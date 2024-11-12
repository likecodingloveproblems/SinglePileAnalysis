from dataclasses import field

counter = 0


def get_tag():
    global counter
    counter += 1
    return counter


class TaggedObject:
    tag: int = field(default_factory=get_tag)
