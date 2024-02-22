from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from utilities.model import get_model

app = FastAPI()
Llama2chat_13b_chain_Window = get_model()


# Define the request model
class ChatRequest(BaseModel):
    question: str


# Define the endpoint
@app.post("/chat")
def retrieve_chat(request: ChatRequest):
    try:
        # Extract data from the request
        question = request.question
        # Use the Conversational Retrieval Chain to retrieve chat
        result =Llama2chat_13b_chain_Window(question)
      
        return {"result": result['answer']}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
