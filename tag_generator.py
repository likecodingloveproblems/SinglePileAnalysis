from pydantic import BaseModel, Field

counter = 0


def get_tag():
    global counter
    counter += 1
    return counter


class TaggedObject(BaseModel):
    tag: int = Field(default_factory=get_tag)
