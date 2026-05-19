from fastapi import FastAPI, HTTPException
from model import predict
from pydantic import BaseModel, field_validator

app = FastAPI(title='Wiyorent Content Moderation Machine Learning System', version='1.0.0')

class ModerateRequest(BaseModel):
    text : str

    @field_validator("text")
    @classmethod
    def text_is_not_empty(cls, v:str) -> str:
        if not v.strip():
            raise ValueError('Text must not be empty')
        return v

        
class ModerateResponse(BaseModel):
    text: str
    label: str
    confidence: float
    scores: dict[str,float]


@app.get('/health')
def health():
    return {'status': 'Doing perfectly fine. Thanks for asking'}

@app.post('/moderate')
def moderate(body: ModerateRequest):
    try: 
        result = predict(body.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ModerateResponse(text=body.text, **result)
