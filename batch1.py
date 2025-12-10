from google import genai

client = genai.Client(
    vertexai=True,
    project="YOUR_PROJECT_ID",
    location="us-central1",
)

def create_batch_inline():

    # Inline items (up to ~20MB total recommended)
    inputs = [
        {
            "id": "req1",
            "model": "google/gemini-2.5-flash-001",
            "input": "Write 5 ideas for an AI startup in healthcare."
        },
        {
            "id": "req2",
            "model": "google/gemini-2.5-flash-001",
            "input": "Explain RAG system for beginners."
        }
    ]

    # Create batch job
    batch = client.batches.create(
        display_name="gemini-flash-batch-inline",
        model="google/gemini-2.5-flash-001",
        input=inputs,     # <-- INLINE INPUT
        output_config={
            "gcs_destination": {
                "uri": "gs://YOUR_BUCKET_NAME/batch_results/"
            }
        }
    )

    print("Batch created:", batch.name)
    return batch


batch = create_batch_inline()
print(batch)
