# %%
"""
This script uploads the API, Examples, and Docs to the vector store.
"""

import os
from pathlib import Path
from openai import OpenAI
from openai import NotFoundError
import git

# get keys
API_KEY = os.environ.get("OPENAI_API_KEY")
VECTOR_STORE_ID = os.environ.get("VECTOR_STORE_ID")

REPO_PATH = Path.cwd()


def detach_and_delete_file(client, vector_store_id, file_id):
    """
    Detach a file from the specified vector store and delete it.

    This function first removes the file from the given vector store,
    then deletes it from the OpenAI account.

    Arguments:
        - client : The OpenAI client instance.
        - vector_store_id : The ID of the vector store.
        - file_id : The ID of the file to detach and delete.

    Returns:
        - None

    Examples:
        ```py
        detach_and_delete_file(client, "store123", "fileABC")
        ```
    """
    try:
        client.vector_stores.files.delete(
            vector_store_id=vector_store_id,
            file_id=file_id,
        )
    except NotFoundError:
        return

    client.files.delete(
        file_id=file_id,
    )


def detach_and_delete_all_files(client, vector_store_id):
    """
    Detach and delete every file in the specified vector store.

    Arguments:
        - client : The OpenAI client instance.
        - vector_store_id : The ID of the vector store.

    Returns:
        - None

    Examples:
        ```py
        detach_and_delete_all_files(client, "storeXYZ")
        ```
    """
    # file_list = client.vector_stores.files.list(
    #     vector_store_id=vector_store_id,
    # )
    file_list = client.files.list(
        limit=10_000,
        order="desc",
    )
    for file in file_list.data:
        try:
            client.vector_stores.files.delete(
                vector_store_id=vector_store_id,
                file_id=file.id,
            )
        except NotFoundError:
            # Skip to the next file if deletion fails due to
            # file not being found
            continue
        client.files.delete(
            file_id=file.id,
        )
    print("All files have been detached and deleted from vector store.")


def upload_and_attach_files(
    client,
    vector_store_id,
    file_path,
):
    """
    Upload a file to the vector store and attach it.

    This function creates the file in OpenAI, then links
    it to the vector store with relevant metadata.

    Arguments:
        - client : The OpenAI client instance.
        - vector_store_id : The ID of the vector store.
        - file_path : Path to the file being uploaded.

    Returns:
        - The vector store response object.

    Examples:
        ```py
        upload_and_attach_files(client, "store123", Path("docs/example.md"))
        ```
    """
    # Upload the file to the vector store
    file_response = client.files.create(
        file=open(file_path, "rb"),  # Open the file in binary mode for upload
        purpose="assistants",  # Specify the purpose of the file
    )

    # create attributes dictionary
    file_name = file_path.name
    full_path = file_path.as_posix()
    path = full_path.split("temp/")[-1]

    attributes = {
        "git_commit": get_current_commit(),
        "file_name": file_name,
        "full_path": full_path,
        "path": path,
    }

    # Attach the uploaded file to the vector store
    return client.vector_stores.files.create(
        vector_store_id=vector_store_id,
        file_id=file_response.id,  # Use the file ID returned from upload
        attributes=attributes,  # Pass the attributes dictionary
    )


def upload_all_files_in_directory(
    client,
    vector_store_id,
    directory_path,
):
    """
    Upload and attach all files in a directory to the vector store.

    Arguments:
        - client : The OpenAI client instance.
        - vector_store_id : The ID of the vector store.
        - directory_path : Path to the folder containing files to upload.

    Returns:
        - None

    Examples:
        ```py
        upload_all_files_in_directory(client, "store123", "C:/temp/docs")
        ```
    """

    folder_dir = Path(directory_path)
    files_list = list(folder_dir.rglob("*.*"))
    # Iterate over all files in the directory
    for file_path in files_list:
        upload_and_attach_files(client, vector_store_id, file_path)
    print(
        f"All files in directory '{directory_path}' have been "
        f"uploaded and attached to vector store."
    )


def get_current_commit():
    """
    Retrieve the current commit hash from the local Git repository.

    Returns:
        - A string with the commit hash.

    Examples:
        ```py
        commit_hash = get_current_commit()
        print(commit_hash)
        ```
    """
    repo = git.Repo(search_parent_directories=True)
    return repo.head.commit.hexsha


def get_changed_files(previous_commit_hash):
    """
    Get a list of files changed between the previous commit and the current
    commit.

    Arguments:
        - previous_commit_hash : The hash of the previous commit.

    Returns:
        - A list of changed file paths.
    """
    repo = git.Repo(search_parent_directories=True)
    previous_commit = repo.commit(previous_commit_hash)
    current_commit = repo.commit("HEAD")
    diff_index = previous_commit.diff(current_commit)

    # Use a set to avoid duplicates (e.g., in renames)
    changed_files = set()
    for diff in diff_index:
        # If a_path is None (e.g., for an added file), use b_path
        if diff.a_path:
            changed_files.add(diff.a_path)
        if diff.b_path:
            changed_files.add(diff.b_path)
    return list(changed_files)


def filter_changed_files(changed_files):
    """
    Filter changed files to include only those in specific directories.

    Arguments:
        - changed_files : List of changed file paths.

    Returns:
        - A list of filtered file paths.
    """
    filtered_files = []
    for file in changed_files:
        if (
            "particula/" in file
            or "docs/Examples" in file
            or "docs/Theory" in file
        ):
            filtered_files.append(file)
    return filtered_files


def collect_files_to_update(client, changed_files):
    """
    Collect files to update based on remote files and changed files.

    Arguments:
        - client : The OpenAI client instance.
        - changed_files : List of changed file paths.

    Returns:
        - A dictionary mapping each changed file's path to update info.

    Examples:
        ```py
        update_dict = collect_files_to_update(client,
            ["particula/foo.py", "docs/Examples/bar.md"])
        ```
    """
    remote_files = client.files.list(
        limit=10_000,
        order="desc",
    )

    # Build a dictionary where keys are base names and values are lists of
    # remote file IDs.
    remote_files_dict = {}
    for file in remote_files.data:
        base_name = file.filename.split(".")[0]
        # Use the base name (without extension) as the key.
        # if the base name already exists, append the file ID to the list.
        # This handles cases where multiple files share the name but diff
        # directories.
        remote_files_dict.setdefault(base_name, []).append(file.id)

    # Build the final dictionary using the precomputed mapping.
    files_to_update = {}
    for file_path in changed_files:
        file_name = file_path.split("/")[-1].split(".")[0]
        if file_name in remote_files_dict:
            files_to_update[file_path] = {
                "file_ids": remote_files_dict[file_name],
                "file_name": file_name,
                "full_path": file_path,
                "method": "update",
            }
        else:
            files_to_update[file_path] = {
                "file_ids": [],  # No matching file IDs found
                "file_name": file_name,
                "full_path": file_path,
                "method": "upload",
            }
    return files_to_update


def upload_new_file(client, file_to_update_dict):
    """
    Upload a new file to the vector store based on the provided info.

    Arguments:
        - client : The OpenAI client instance.
        - file_to_update_dict : Dictionary containing file info (path, etc.).

    Returns:
        - None

    Examples:
        ```py
        upload_new_file(client, {"full_path": "docs/example.md",
            "method": "upload", ...})
        ```

    """
    # Create markdown path based on the file's base name
    markdown_path = REPO_PATH / "docs/.assets/temp" / (
        file_to_update_dict["full_path"].split(".")[0] + ".md"
    )
    # Check if the markdown file exists and upload it if it does
    if markdown_path.exists():
        upload_and_attach_files(
            client=client,
            vector_store_id=VECTOR_STORE_ID,
            file_path=markdown_path,
        )
        print(
            f"Markdown file for '{file_to_update_dict["full_path"]}' "
            " has been uploaded and attached to the vector store."
        )
    else:
        print(
            f"Markdown file for '{file_to_update_dict["full_path"]}' "
            "does not exist. Skipping upload."
        )


def update_vector_store_file(client, file_to_update_dict):
    """
    Update a file in the vector store based on its existing ID.

    Arguments:
        - client : The OpenAI client.
        - file_to_update_dict :Dictionary containing file metadata:
            - "file_ids" : Remote file IDs
            - "full_path" : Local file path
            - "file_name" : Base name
            - "method" : "update" or "upload"

    Returns:
        - None

    Examples:
        ```py
        update_vector_store_file(client, {"file_ids": [...],
            "full_path": "docs/test.md", ...})
        ```
    """
    # Iterate over remote file IDs
    for file_id in file_to_update_dict["file_ids"]:
        vector_response = client.vector_stores.files.retrieve(
            vector_store_id=VECTOR_STORE_ID,
            file_id=file_id,
        )

        # Check if the path matches based on the base filename
        path_match = (
            vector_response.attributes["path"].split(".")[0]
            == file_to_update_dict["full_path"].split(".")[0]
        )
        if path_match:
            detach_and_delete_file(
                client=client,
                vector_store_id=VECTOR_STORE_ID,
                file_id=vector_response.id,
            )
            upload_new_file(
                client=client,
                file_to_update_dict=file_to_update_dict,
            )


def set_vector_store_commit(
    client,
    vector_store_id,
    commit_hash=None,
):
    """
    Set the vector store's commit hash to the current local commit.

    Arguments:
        - client : The OpenAI client instance.
        - vector_store_id : The ID of the vector store.

    Returns:
        - None
    """
    if commit_hash is None:
        # Get the current commit hash
        commit_hash = get_current_commit()

    # Update the vector store with the commit hash
    client.vector_stores.update(
        vector_store_id=vector_store_id,
        metadata={
            "git_commit": commit_hash,
        },
    )
    print(f"Vector store commit hash has been set to: {commit_hash}")


def get_vector_store_commit(client, vector_store_id):
    """
    Retrieve the commit hash stored in the vector store's metadata.

    Arguments:
        - client : The OpenAI client instance.
        - vector_store_id : The ID of the vector store.

    Returns:
        - A string with the commit hash.

    Examples:
        ```py
        current_vs_commit = get_vector_store_commit(client, "store123")
        ```
    """
    vector_store = client.vector_stores.retrieve(
        vector_store_id=vector_store_id,
    )
    return vector_store.metadata["git_commit"]


def refresh_changed_files(client, vector_store_id):
    """
    Refresh changed files in the vector store based on local Git diffs.

    Arguments:
        - client : The OpenAI client instance.
        - vector_store_id : The ID of the vector store.

    Returns:
        - None

    Examples:
        ```py
        refresh_changed_files(client, "storeXYZ")
        ```
    """
    # prev_commit = "428ab0e02319e0254e7d7532e36fb32403293d51"  # testing
    prev_commit = get_vector_store_commit(  # production
        client=client,
        vector_store_id=vector_store_id,
    )
    print(f"Previous commit hash: {prev_commit}")
    changed_files = get_changed_files(prev_commit)
    changed_files = filter_changed_files(changed_files)

    file_to_update = collect_files_to_update(
        client=client,
        changed_files=changed_files,
    )
    print(f"Files to update: {file_to_update.keys()}")

    # Check if there are any files to update
    if not file_to_update:
        print("No files to update.")
        return
    # Iterate over the files to update
    for file_path, file_info in file_to_update.items():
        if file_info["method"] == "update":
            update_vector_store_file(
                client=client,
                file_to_update_dict=file_info,
            )
        elif file_info["method"] == "upload":
            upload_new_file(
                client=client,
                file_to_update_dict=file_info,
            )


def refresh_all_files(
    client,
    vector_store_id=VECTOR_STORE_ID,
):
    """
    Delete all files from the vector store and re-upload everything from temp.

    Arguments:
        - client : The OpenAI client instance.

    Returns:
        - None

    Examples:
        ```py
        refresh_all_files(client)
        ```
    """
    detach_and_delete_all_files(
        client=client,
        vector_store_id=vector_store_id,
    )
    upload_all_files_in_directory(
        client=client,
        vector_store_id=vector_store_id,
        directory_path=(REPO_PATH / "docs/.assets/temp"),
        # Path to the directory containing files to upload
    )


# %%
# -------------------------------
# Main function to run the script
# -------------------------------


if __name__ == "__main__":

    client = OpenAI(api_key=API_KEY)

    # set commit hash to the vector store
    # set_vector_store_commit(
    #     client=client,
    #     vector_store_id=VECTOR_STORE_ID,
    #     commit_hash="4d8de48a322c9d683a34c35ec7b6897cf27945af",
    # )

    # print current commit hash
    commit_hash = get_current_commit()
    print(f"Current commit hash: {commit_hash}")

    refresh_changed_files(
        client=client,
        vector_store_id=VECTOR_STORE_ID,
    )

    # Uncomment the following line to refresh all files
    # refresh_all_files(
    #     client=client,
    #     vector_store_id=VECTOR_STORE_ID,
    # )

    # set_vector_store_commit(
    #     client=client,
    #     vector_store_id=VECTOR_STORE_ID,
    # )
    print("Vector store update check is finished.")

# %%
