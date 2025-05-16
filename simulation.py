import asyncio
import json
import networkx as nx
import os
import random
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import logging

# Updated import for Google Generative AI
from google import genai

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load API Key from secrets file
try:
    with open(".secrets.json", "r") as f:
        secrets = json.load(f)
    API_KEY = secrets["API_KEY"]
except FileNotFoundError:
    raise RuntimeError("Please create a .secrets.json file with your API_KEY")
except KeyError:
    raise RuntimeError(".secrets.json file must contain an 'API_KEY' entry")

# Initialize the Gemini client with your API key
client = genai.Client(api_key=API_KEY)

# Configure available models and their parameters
MODEL_CONFIGS = {
    "gemini-1.5-flash": {
        "max_batch_size": 5,  # Adjust based on your quota and performance needs
        "retry_delay": 2,     # Seconds to wait between retries
        "max_retries": 3      # Maximum number of retries on failure
    },
    "gemini-2.0-flash": {
        "max_batch_size": 3,
        "retry_delay": 2,
        "max_retries": 3
    },
    "gemini-1.5-pro": {
        "max_batch_size": 2,
        "retry_delay": 3,
        "max_retries": 3
    }
}

# Default model
DEFAULT_MODEL = "gemini-1.5-flash"

@dataclass
class Message:
    text: str
    origin: str  # node ID or "official"
    truth: bool  # true/false claim

def compute_belief(agent: dict, msg: Message) -> float:
    base = agent['trust_official'] if msg.origin == 'official' else agent['susceptibility']
    return base * (0.8 if msg.text in agent['memory'] else 1.0)

def compute_share(agent: dict, belief_score: float) -> bool:
    return belief_score >= agent['share_threshold']

# Build new graph
def build_graph(
    n: int = 150, k: int = 4, p: float = 0.1,
    share_threshold: float = 0.5
) -> nx.Graph:
    G = nx.watts_strogatz_graph(n=n, k=k, p=p)
    for node in G.nodes():
        G.nodes[node].update({
            'trust_official': random.random(),
            'susceptibility': random.random(),
            'share_threshold': share_threshold,
            'memory': set(),
        })
    return G

async def generate_post_with_retry(agent_id: int, message_text: str, model: str = DEFAULT_MODEL) -> str:
    """Generate a post with retry logic for API calls"""
    config = MODEL_CONFIGS.get(model, MODEL_CONFIGS[DEFAULT_MODEL])
    max_retries = config["max_retries"]
    retry_delay = config["retry_delay"]
    
    prompt = (
        f"You are user #{agent_id} who BELIEVES the following claim: \"{message_text}\".\n"
        "Write a social media post to share this information in your own style."
    )
    
    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt
            )
            return response.text
        except Exception as e:
            if attempt < max_retries:
                logger.warning(f"Attempt {attempt+1} failed for agent {agent_id}: {str(e)}. Retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
            else:
                logger.error(f"All attempts failed for agent {agent_id}: {str(e)}")
                # Return a generic message if all attempts fail
                return f"I heard that {message_text}"

async def batch_generate_posts(agent_msg_pairs: List[Tuple[int, str]], model: str = DEFAULT_MODEL) -> List[str]:
    """Generate posts in efficient batches to avoid rate limiting"""
    config = MODEL_CONFIGS.get(model, MODEL_CONFIGS[DEFAULT_MODEL])
    max_batch_size = config["max_batch_size"]
    
    results = []
    for i in range(0, len(agent_msg_pairs), max_batch_size):
        batch = agent_msg_pairs[i:i+max_batch_size]
        tasks = [generate_post_with_retry(agent_id, msg, model) for agent_id, msg in batch]
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)
        
        # Add a small delay between batches to avoid rate limiting
        if i + max_batch_size < len(agent_msg_pairs):
            await asyncio.sleep(0.5)
    
    return results

# Run the T-step simulation and return cumulation believers each step
async def run_simulation_async(
    G: nx.Graph,
    seed_text: str,
    correction_text: str,
    T: int = 30,
    model: str = DEFAULT_MODEL
) -> list[int]:
    
    # Seed the rumor
    seed = Message(seed_text, origin="botnet", truth=False)
    inboxes = {n: [] for n in G.nodes()}
    for s in random.sample(list(G.nodes()), k=3):
        inboxes[s].append(seed)
        G.nodes[s]['memory'].add(seed_text)

    believers_over_time = []
    for t in range(T):
        logger.info(f"Running simulation step {t+1}/{T}")
        new_inboxes = {n: [] for n in G.nodes()}
        share_list = []

        # Agents receive messages and decide whether to share
        for node in G.nodes():
            agent = G.nodes[node]
            for msg in inboxes[node]:
                b = compute_belief(agent, msg)
                agent['memory'].add(msg.text)
                if compute_share(agent, b):
                    share_list.append((node, msg))
        
        # Agents share messages
        if share_list:
            logger.info(f"Step {t+1}: {len(share_list)} agents sharing messages")
            
            # Prepare input for batch generation
            agent_msg_pairs = [(int(node), msg.text) for node, msg in share_list]
            
            # Generate posts in batches
            try:
                texts = await batch_generate_posts(agent_msg_pairs, model)
                
                # Process the results
                for (node, msg), text in zip(share_list, texts):
                    new_msg = Message(
                        text, 
                        origin=node, 
                        truth=msg.truth
                    )
                    for nbr in G.neighbors(node):
                        new_inboxes[nbr].append(new_msg)
            except Exception as e:
                logger.error(f"Error in batch generation: {str(e)}")

        # Agents receive correction messages
        if t == 1:
            correction = Message(
                correction_text, 
                origin="official", 
                truth=True
            )
            for n in G.nodes():
                if G.nodes[n]['trust_official'] > 0.7:
                    new_inboxes[n].append(correction)
                    G.nodes[n]['memory'].add(correction_text)

        # Update inboxes
        inboxes = new_inboxes
        cumul = sum(1 for n in G.nodes() if seed_text in G.nodes[n]['memory'])
        believers_over_time.append(cumul)
        logger.info(f"Step {t+1} complete: {cumul}/{len(G.nodes())} believers")
    
    return believers_over_time

def run_simulation(
    graph_size: int,
    seed_text: str,
    correction_text: str,
    T: int,
    share_threshold: float,
    model: str = DEFAULT_MODEL
) -> list[int]:
    """
    Run the misinformation spread simulation
    
    Parameters:
    - graph_size: Number of nodes in the social network
    - seed_text: The initial misinformation to spread
    - correction_text: The official correction message
    - T: Number of time steps to simulate
    - share_threshold: Threshold for agents to share messages
    - model: AI model to use for text generation (default: "gemini-1.5-flash")
    
    Returns:
    - List of cumulative believers at each time step
    """
    # Validate model choice
    if model not in MODEL_CONFIGS:
        logger.warning(f"Model '{model}' not recognized. Using default model '{DEFAULT_MODEL}'")
        model = DEFAULT_MODEL
    
    logger.info(f"Building graph with {graph_size} nodes")
    G = build_graph(
        n=graph_size, 
        share_threshold=share_threshold
    )
    
    logger.info(f"Starting simulation with model: {model}")
    return asyncio.run(run_simulation_async(G, seed_text, correction_text, T, model))