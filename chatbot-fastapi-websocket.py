from fastapi import FastAPI, WebSocket
import asyncio
from llm_utils import generate_response # Assume this is a function that calls an LLM
import websocket
import json

app = FastAPI()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections = {}

    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections[user_id] = websocket

    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            del self.active_connections[user_id]

    async def send_to_user(self, user_id: str, data: dict):
        if user_id in self.active_connections:
            await self.active_connections[user_id].send_json(data)

# WebSocket endpoint for chatbot
connection_manager = ConnectionManager()

@app.websocket("/chatbot/{user_id}")
async def chatbot_websocket(websocket: WebSocket, user_id: str):
    await connection_manager.connect(websocket, user_id)
    try:
        async for message in websocket.iter_text():
            # Process the user's message
            user_input = message
            
            # Generate the first response using the LLM
            response1 = await generate_response(user_input)
            
            # Send the first response immediately
            await connection_manager.send_to_user(user_id, {"response1": response1})
            
            # Generate the second response using the LLM
            response2 = await generate_response(user_input)
            
            # Send the second response
            await connection_manager.send_to_user(user_id, {"response2": response2})
    except:
        connection_manager.disconnect(user_id)

# Jupyter Notebook client example
import websocket
import json

def on_message(ws, message):
    data = json.loads(message)
    print(f"Response 1: {data['response1']}")
    print(f"Response 2: {data['response2']}")

def on_error(ws, error):
    print(f"WebSocket error: {error}")

def on_close(ws):
    print("WebSocket connection closed")

def on_open(ws):
    ws.send("Hello from Jupyter Notebook!")

def test_chatbot_websocket():
    ws = websocket.WebSocketApp(
        "ws://localhost:8000/chatbot/test_user",
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.on_open = on_open
    ws.run_forever()

test_chatbot_websocket()
```

In this updated example, we've added a Jupyter Notebook client example at the bottom of the artifact.

The key changes are:

1. We've added a `test_chatbot_websocket()` function that creates a WebSocketApp instance and connects to the `/chatbot/test_user` endpoint.
2. The `on_message`, `on_error`, `on_close`, and `on_open` callbacks are defined to handle the different WebSocket events.
   - `on_message` is used to process the incoming responses from the server.
   - `on_error` and `on_close` handle WebSocket connection errors and closures.
   - `on_open` is used to send an initial message to the server when the connection is established.
3. The `test_chatbot_websocket()` function is called to start the WebSocket client and interact with the FastAPI WebSocket API.

To test this, you can run the FastAPI server and then execute the `test_chatbot_websocket()` function in your Jupyter Notebook. This will establish a WebSocket connection to the server, send a message, and display the responses received from the server.

Using a Jupyter Notebook for testing allows you to easily interact with the WebSocket API, experiment with different inputs, and debug the communication between the client and the server. This can be particularly helpful during the development and testing phases of your chatbot application.

Let me know if you have any other questions!
