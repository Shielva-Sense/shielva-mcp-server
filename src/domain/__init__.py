"""Domain layer — pure business logic, NO framework dependencies.

Files under ``domain/`` MUST NOT import:
    * fastapi, pydantic.BaseModel for response shapes
      (Pydantic dataclasses for value-objects are fine)
    * sqlalchemy, motor, pymongo
    * httpx, litellm, openai
    * structlog (use stdlib logging if needed)

Domain code expresses *what the business does*, not how. Adapters
under ``infrastructure/`` implement the ports (interfaces) defined
here. Application services under ``application/`` orchestrate them.

This is the hexagonal/clean-architecture inner ring.
"""
