from fastapi import FastAPI
# import dotenv for env files
from dotenv import load_dotenv
import os

# import openai
import openai

# model creation
from pydantic import BaseModel
from typing import Union

load_dotenv()

# create model
class SimpleText(BaseModel):
    prompt: str


class TextWithInstruction(BaseModel):
    input: str
    prompt: str


# create app instance
app = FastAPI()

openai.api_key = os.getenv("OPENAI_API_KEY")


@app.post("/")
def text_completion(simple_text: SimpleText):
    response = openai.Completion.create(
        model="text-davinci-003", prompt=simple_text.prompt, temperature=0, max_tokens=7)
    return {"response": response}


@app.post("/instructions")
def text_with_instruction(text: TextWithInstruction):
    response = openai.Edit.create(
        model="text-davinci-edit-001", input=text.input, instruction=text.prompt, temperature=0)
    return {"response": response}


@app.post("/images")
def text_with_instruction(text: SimpleText):
    response = openai.Image.create(
        prompt=text.prompt,
        n=1,
        size="1024x1024"
    )
    image_url = response['data'][0]['url']

    return {"response": image_url}


@app.post("/sentiment_analysis")
def text_with_instruction(text: SimpleText):
    response = openai.Moderation.create(
        input=text.prompt
    )
    sentiment_report = response["results"][0]

    return {"response": sentiment_report}
