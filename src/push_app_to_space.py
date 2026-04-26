from huggingface_hub import upload_folder


HF_USERNAME = "Yuvisukumar"
HF_SPACE_REPO = f"{HF_USERNAME}/superkart-sales-app"


def push_app_to_space():
    upload_folder(
        folder_path="app",
        repo_id=HF_SPACE_REPO,
        repo_type="space"
    )

    print("Streamlit app files pushed to Hugging Face Space.")


if __name__ == "__main__":
    push_app_to_space()