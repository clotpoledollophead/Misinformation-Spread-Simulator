import asyncio
import google.genai as genai
import json
import networkx as nx
import os
import random

from dataclasses import dataclass
# from openai import OpenAI

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("Please set the API_KEY environment variable")
client = genai(api_key=API_KEY)

@dataclass
class Message:
    text: str
    origin: str # node ID or "official"
    truth: bool # true/false claim

def compute_belief(agent: dict, msg: Message) -> float:
    base = agent['trust_official'] if msg.origin == 'official' else agent['susceptibility']
    return base * (0.8 if msg.text in agent['memory'] else 1.0)

def compute_share(agent: dict, belief_score: float) -> bool:
    return belief_score >= agent['share_threshold']

# build new graph
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

def _sync_generate_post(agent_id: int, message_text: str) -> str:
    prompt = (
        f"You are user #{agent_id} who BELIEVES the following claim: \"{message_text}\".\n"
        "Write a social media post to share this information in you rown style."
    )

    resp = genai.chat.create(
        # model="gpt-3.5-turbo",
        model = "gemma-3-27b-it",
        temperature=0.7,
        candidates=1,
        messages=[{"role": "user",
                   "content": prompt}],
                   temperature=0.7
    )

    return resp.candidates[0].content


async def generate_post(agent_id: int, message_text: str) -> str:
    # async wrapper so it doesn’t block the event loop
    return await asyncio.to_thread(_sync_generate_post, agent_id, message_text)

# run the T-step simulation and return cumulation believers each step
async def run_simulation_async(
    G: nx.Graph,
    seed_text: str,
    correction_text: str,
    T: int = 30
) -> list[int]:
    
    # seed the rumor
    seed = Message(seed_text, origin="botnet", truth=False)
    inboxes = {n: [] for n in G.nodes()}
    for s in random.sample(list(G.nodes()), k=3):
        inboxes[s].append(seed)
        G.nodes[s]['memory'].add(seed_text)

    believers_over_time = []
    for t in range(T):
        new_inboxes = {n: [] for n in G.nodes()}
        share_list   = []

        # agents receive messages and decide whether to share
        for node in G.nodes():
            agent = G.nodes[node]
            for msg in inboxes[node]:
                b = compute_belief(agent, msg)
                agent['memory'].add(msg.text)
                if compute_share(agent, b):
                    share_list.append((node, msg))
        
        # agents share messages
        if share_list:
            tasks = [generate_post(n, m.text) for n, m in share_list]
            texts = await asyncio.gather(*tasks)
            for (node, msg), text in zip(share_list, texts):
                new_msg = Message(
                    text, 
                    origin=node, 
                    truth=msg.truth
                    )
                for nbr in G.neighbors(node):
                    new_inboxes[nbr].append(new_msg)

        # agents receive correction messages
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

        # update inboxes
        inboxes = new_inboxes
        cumul = sum(1 for n in G.nodes() if seed_text in G.nodes[n]['memory'])
        believers_over_time.append(cumul)
    
    return believers_over_time

def run_simulation(
    graph_size: int,
    seed_text: str,
    correction_text: str,
    T: int,
    share_threshold: float
) -> list[int]:
    G = build_graph(
        n=graph_size, 
        share_threshold=share_threshold
        )
    return asyncio.run(run_simulation_async(G, seed_text, correction_text, T))

