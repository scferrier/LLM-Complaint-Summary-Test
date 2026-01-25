from qdrant_client import QdrantClient

qdrant_client = QdrantClient(
    url="https://b2746187-c6b8-46f8-868a-71a7650aa08c.us-east4-0.gcp.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.TysGipT0UWEhwHV3Au1jDPqtN7vmbP-o0XYrq9q4qnM",
)

print(qdrant_client.get_collections())