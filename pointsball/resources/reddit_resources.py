from dagster import ConfigurableResource


class RedditResource(ConfigurableResource):
    client_id: str
    secret: str
    user_agent: str
    username: str
    password: str
