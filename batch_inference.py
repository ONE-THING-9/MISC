from google.cloud import aiplatform
import json
import base64

PROJECT_ID = "YOUR_PROJECT_ID"
LOCATION = "us-central1"       # or other region where Gemini batch is supported
MODEL_NAME = "gemini-2.5-flash-001"   # your Gemini model

def create_batch_job_inline():

    aiplatform.init(project=PROJECT_ID, location=LOCATION)

    # ---------------------------
    # 1. Build inline JSONL input
    # ---------------------------
    instances = [
        {
            "prompt": "Write 5 startup ideas based on AI + healthcare."
        },
        {
            "prompt": "Explain quantum computing like I'm 10 years old."
        }
    ]

    # Convert each instance into JSONL format
    jsonl_content = "\n".join([json.dumps(i) for i in instances])

    # Encode inline input as base64 (Vertex Batch API requirement)
    inline_data_item = {
        "inline_source": {
            "files": [
                {
                    "file_name": "input.jsonl",
                    "file_bytes": base64.b64encode(
                        jsonl_content.encode("utf-8")
                    ).decode("utf-8"),
                }
            ]
        }
    }

    # ---------------------------
    # 2. Create and run batch job
    # ---------------------------
    batch_prediction_job = aiplatform.BatchPredictionJob.create(
        display_name="gemini_inline_batch_job",
        model=f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL_NAME}",
        input_config=inline_data_item,
        output_config={
            "gcs_destination": {
                "output_uri_prefix": f"gs://YOUR_BUCKET/gemini_output/"
            }
        },
        generate_explanation=False,
        dedicated_resources={
            "machine_type": "e2-standard-16",
        }
    )

    print("Batch Job Created:")
    print(batch_prediction_job.resource_name)
    return batch_prediction_job


if __name__ == "__main__":
    job = create_batch_job_inline()
    print("Job state:", job.state)
