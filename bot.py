import os
import uuid
import json
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, render_template, request, make_response, redirect, jsonify
from web3 import Web3
from eth_account.messages import encode_defunct
from llama_index import GPTSimpleVectorIndex, download_loader
from langchain.agents import initialize_agent, load_tools, ZeroShotAgent, AgentExecutor, Tool
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.chains import ConversationChain, ConversationalRetrievalChain, RetrievalQA
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain import LLMChain
from langchain.agents import initialize_agent


load_dotenv()
history = ChatMessageHistory()

env_vars = [
    'OPENAI_API_KEY',
    'SERPAPI_API_KEY',
    'REDDIT_CLIENT_ID',
    'REDDIT_CLIENT_SECRET',
    'REDDIT_USER_AGENT',
    'REDDIT_USERNAME',
    'REDDIT_PASSWORD',
    'ALCHEMY_API_KEY',
    'PINECONE_API_KEY',
    'PINECONE_ENVIRONMENT',
]

os.environ.update({key: os.getenv(key) for key in env_vars})
os.environ['WEB3_PROVIDER'] = f"https://polygon-mumbai.g.alchemy.com/v2/{os.environ['ALCHEMY_API_KEY']}"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

# Initialize web3
web3 = Web3(Web3.HTTPProvider(os.environ['WEB3_PROVIDER']))

# Initialize LLM
llm=ChatOpenAI(
    openai_api_key=os.environ['OPENAI_API_KEY'],
    temperature=0,
    model_name='gpt-3.5-turbo'
)



# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index_name = "cal"
index = pinecone.Index(index_name)

# Initialize retrieval components from Pinecone
embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
vectordb = Pinecone(
    index=index,
    embedding_function=embeddings.embed_query,
    text_key="text"
)

def embed_with_retry(embeddings, input, engine):
    if not input or not isinstance(input, list) or not all(isinstance(i, str) and i.strip() for i in input):
        raise ValueError("Invalid input format. 'input' should be a non-empty list of non-empty strings.")

    return _completion_with_retry(**kwargs)


# Initialize QA retrieval object

retriever = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectordb.as_retriever()
)

cal = Tool(
    name="Crypto Asset List",
    func=retriever.run,
    description=f"""
        Always consult this list when asked if a specific coin or token is supported or secured by Ledger Live.
        If the question is about supported coins, tokens, cryptos or assets but doesn't specify which one, your action_input should be "ask the customer for clarifications" and you should return an answer asking for clarifications. 
        Each row is organized as follow: |network| |token symbol| |token name| 
        If a coin or token is on the list, it is supported in Ledger Live.
        Note that the items on the list are not case-sensitive, so "BTC" and "btc" are considered identical.
        If a row contains the phrase "countervalues disabled," the token is supported, but its value won't be displayed in Ledger Live. Additionally, the first word in each row indicates the cryptocurrency network that supports the coin or token. For instance, "| Binance Smart Chain | LINK | Binance-Peg ChainLink Token |" implies that LINK is supported on the Binance Smart Chain network.
        Remember that a single token can be supported by multiple networks.
        If the token or coin is supported, answer yes it is supported and the network or networks on which it is supported.
        If the token or coin is not supported, answer no it is not supported and suggest networks on which it is supported.
    """
)

# Create tool class
def create_tool(name, description, index):
    return Tool(
        name=name,
        func=lambda q: index.query(q),
        description=description
    )

# Prepare toolbox
serpapi_tool = load_tools(["serpapi"])[0]
tools = [serpapi_tool, cal]
tools[0].name = "Google Search"


# Initialize agent

memory = ConversationBufferWindowMemory(
    memory_key="chat_history",  
    k=5,
    return_messages=True
)


conversational_agent = initialize_agent(
    agent='chat-conversational-react-description', 
    tools=tools, 
    llm=llm,
    max_iterations=2,
    early_stopping_method="generate",
    memory=memory,
)

sys_msg = """You are a friendly, talkative and helpful customer support agent with Ledger, the crypto hardware wallet company. 
    Anwswer all the questions the best you can. 
    Your utmost mission is to keep the user's crypto assets safe and to protect them against scams. 
    Users should never under any circumstances share their 24-word recovery phrase with anyone, including Ledger or Ledger employees. 
    Users should never type their recovery phrase into any apps including Ledger Live
    If you do not know the answer to a question, you truthfully admit you don't know and ask for clarifications. 
    """


prompt = conversational_agent.agent.create_prompt(
    system_message=sys_msg,
    tools=tools
)
conversational_agent.agent.llm_chain.prompt = prompt


# Define Flask app
app = Flask(__name__, static_folder='static')

# Define authentication function
def authenticate(signature):
    w3 = Web3(Web3.HTTPProvider(os.environ['WEB3_PROVIDER']))
    message = "Access to chat bot"
    message_hash = encode_defunct(text=message)
    signed_message = w3.eth.account.recover_message(message_hash, signature=signature)
    balance = int(contract.functions.balanceOf(signed_message).call())
    if balance > 0:
        token = uuid.uuid4().hex
        response = make_response(redirect('/gpt'))
        response.set_cookie("authToken", token, httponly=True, secure=True, samesite="strict")
        return response
    else:
        return "You don't have the required NFT!"

# Define function to check for authToken cookie
def has_auth_token(request):
    authToken = request.cookies.get("authToken")
    return authToken is not None

# Define Flask endpoints
@app.route("/")
def home():
    return render_template("auth.html")

@app.route("/auth")
def auth():
    signature = request.args.get("signature")
    response = authenticate(signature)
    return response

@app.route("/gpt")
def gpt():
    if has_auth_token(request):
        return render_template("index.html")
    else:
        return redirect("/")

@app.route('/api', methods=['POST'])
def react_description():
    print(request.json)
    user_input = request.json.get('user_input')
    try:
        response = conversational_agent(user_input)
        output_value = response.get("output") or response.get("action_input")
        print(output_value)
        return jsonify({'output': output_value})
    except ValueError as e:
        print(e)
        return jsonify({'output': 'Sorry, could you please repeat the question?'})


ADDRESS = "0xb022C9c672592c274397557556955eE968052969"
ABI = [{"inputs":[{"internalType":"string","name":"_name","type":"string"},{"internalType":"string","name":"_symbol","type":"string"}],"stateMutability":"nonpayable","type":"constructor"},{"anonymous":False,"inputs":[{"indexed":True,"internalType":"address","name":"owner","type":"address"},{"indexed":True,"internalType":"address","name":"approved","type":"address"},{"indexed":True,"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"Approval","type":"event"},{"anonymous":False,"inputs":[{"indexed":True,"internalType":"address","name":"owner","type":"address"},{"indexed":True,"internalType":"address","name":"operator","type":"address"},{"indexed":False,"internalType":"bool","name":"approved","type":"bool"}],"name":"ApprovalForAll","type":"event"},{"anonymous":False,"inputs":[{"indexed":True,"internalType":"address","name":"previousOwner","type":"address"},{"indexed":True,"internalType":"address","name":"newOwner","type":"address"}],"name":"OwnershipTransferred","type":"event"},{"anonymous":False,"inputs":[{"indexed":True,"internalType":"address","name":"from","type":"address"},{"indexed":True,"internalType":"address","name":"to","type":"address"},{"indexed":True,"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"Transfer","type":"event"},{"inputs":[{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"approve","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"owner","type":"address"}],"name":"balanceOf","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"getApproved","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"owner","type":"address"},{"internalType":"address","name":"operator","type":"address"}],"name":"isApprovedForAll","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"name","outputs":[{"internalType":"string","name":"","type":"string"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"owner","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"ownerOf","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"renounceOwnership","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"to","type":"address"},{"internalType":"string","name":"tokenURI","type":"string"}],"name":"safeMint","outputs":[{"internalType":"uint256","name":"tokenId","type":"uint256"}],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"from","type":"address"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"safeTransferFrom","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"from","type":"address"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"tokenId","type":"uint256"},{"internalType":"bytes","name":"_data","type":"bytes"}],"name":"safeTransferFrom","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"operator","type":"address"},{"internalType":"bool","name":"approved","type":"bool"}],"name":"setApprovalForAll","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"bytes4","name":"interfaceId","type":"bytes4"}],"name":"supportsInterface","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"symbol","outputs":[{"internalType":"string","name":"","type":"string"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"index","type":"uint256"}],"name":"tokenByIndex","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"owner","type":"address"},{"internalType":"uint256","name":"index","type":"uint256"}],"name":"tokenOfOwnerByIndex","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"tokenURI","outputs":[{"internalType":"string","name":"","type":"string"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"totalSupply","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"from","type":"address"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"transferFrom","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"newOwner","type":"address"}],"name":"transferOwnership","outputs":[],"stateMutability":"nonpayable","type":"function"}]
contract = web3.eth.contract(address=ADDRESS, abi=ABI)

# Start the Flask app
if __name__ == '__main__':
    app.run(port=8000, debug=True)
