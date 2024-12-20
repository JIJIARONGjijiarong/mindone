"""convert pytorch_model.bin to model.safetensors"""

import argparse
import os
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import torch
from huggingface_hub import CommitInfo, CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub.file_download import repo_folder_name
from safetensors.torch import _find_shared_tensors, _is_complete, load_file, save_file

ConversionResult = Tuple[List["CommitOperationAdd"], List[Tuple[str, "Exception"]]]


def _remove_duplicate_names(
    state_dict: Dict[str, torch.Tensor],
    *,
    preferred_names: List[str] = None,
    discard_names: List[str] = None,
) -> Dict[str, List[str]]:
    if preferred_names is None:
        preferred_names = []
    preferred_names = set(preferred_names)
    if discard_names is None:
        discard_names = []
    discard_names = set(discard_names)

    shareds = _find_shared_tensors(state_dict)
    to_remove = defaultdict(list)
    for shared in shareds:
        complete_names = set([name for name in shared if _is_complete(state_dict[name])])
        if not complete_names:
            if len(shared) == 1:
                # Force contiguous
                name = list(shared)[0]
                state_dict[name] = state_dict[name].clone()
                complete_names = {name}
            else:
                raise RuntimeError(
                    f"Error while trying to find names to remove to save state dict, but found no suitable name to keep for saving amongst: {shared}."
                )

        keep_name = sorted(list(complete_names))[0]

        # Mecanism to preferentially select keys to keep
        # coming from the on-disk file to allow
        # loading models saved with a different choice
        # of keep_name
        preferred = complete_names.difference(discard_names)
        if preferred:
            keep_name = sorted(list(preferred))[0]

        if preferred_names:
            preferred = preferred_names.intersection(complete_names)
            if preferred:
                keep_name = sorted(list(preferred))[0]
        for name in sorted(shared):
            if name != keep_name:
                to_remove[keep_name].append(name)
    return to_remove


def get_discard_names(model_id: str, revision: Optional[str], folder: str, token: Optional[str]) -> List[str]:
    try:
        import json

        import transformers

        config_filename = hf_hub_download(
            model_id, revision=revision, filename="config.json", token=token, cache_dir=folder
        )
        with open(config_filename, "r") as f:
            config = json.load(f)
        architecture = config["architectures"][0]

        class_ = getattr(transformers, architecture)

        # Name for this varible depends on transformers version.
        discard_names = getattr(class_, "_tied_weights_keys", [])

    except Exception:
        discard_names = []
    return discard_names


class AlreadyExists(Exception):
    pass


def check_file_size(sf_filename: str, pt_filename: str):
    sf_size = os.stat(sf_filename).st_size
    pt_size = os.stat(pt_filename).st_size

    if (sf_size - pt_size) / pt_size > 0.01:
        raise RuntimeError(
            f"""The file size different is more than 1%: \n- {sf_filename}: {sf_size} \n- {pt_filename}: {pt_size}"""
        )


def rename(pt_filename: str) -> str:
    filename, ext = os.path.splitext(pt_filename)
    local = f"{filename}.safetensors"
    local = local.replace("pytorch_model", "model")
    return local


def convert_single(
    model_id: str, *, revision: Optional[str], folder: str, token: Optional[str], discard_names: List[str]
) -> ConversionResult:
    pt_filename = hf_hub_download(
        repo_id=model_id, revision=revision, filename="pytorch_model.bin", token=token, cache_dir=folder
    )

    sf_name = "model.safetensors"
    sf_filename = os.path.join(folder, sf_name)
    convert_file(pt_filename, sf_filename, discard_names)
    operations = [CommitOperationAdd(path_in_repo=sf_name, path_or_fileobj=sf_filename)]
    errors: List[Tuple[str, "Exception"]] = []
    return operations, errors


def convert_file(
    pt_filename: str,
    sf_filename: str,
    discard_names: List[str],
):
    loaded = torch.load(pt_filename, map_location="cpu")
    if "state_dict" in loaded:
        loaded = loaded["state_dict"]
    to_removes = _remove_duplicate_names(loaded, discard_names=discard_names)

    metadata = {"format": "pt"}
    for kept_name, to_remove_group in to_removes.items():
        for to_remove in to_remove_group:
            if to_remove not in metadata:
                metadata[to_remove] = kept_name
            del loaded[to_remove]
    # Force tensors to be contiguous
    loaded = {k: v.contiguous() for k, v in loaded.items()}

    dirname = os.path.dirname(sf_filename)
    os.makedirs(dirname, exist_ok=True)
    save_file(loaded, sf_filename, metadata=metadata)
    check_file_size(sf_filename, pt_filename)
    reloaded = load_file(sf_filename)
    for k in loaded:
        pt_tensor = loaded[k]
        sf_tensor = reloaded[k]
        if not torch.equal(pt_tensor, sf_tensor):
            raise RuntimeError(f"The output tensors do not match for key {k}")


def convert_generic(
    model_id: str, *, revision=Optional[str], folder: str, filenames: Set[str], token: Optional[str]
) -> ConversionResult:
    operations = []
    errors = []

    extensions = set([".bin", ".ckpt"])
    for filename in filenames:
        prefix, ext = os.path.splitext(filename)
        if ext in extensions:
            pt_filename = hf_hub_download(model_id, revision=revision, filename=filename, token=token, cache_dir=folder)
            dirname, raw_filename = os.path.split(filename)
            if raw_filename == "pytorch_model.bin":
                sf_in_repo = os.path.join(dirname, "model.safetensors")
            else:
                sf_in_repo = f"{prefix}.safetensors"
            sf_filename = os.path.join(folder, sf_in_repo)
            try:
                convert_file(pt_filename, sf_filename, discard_names=[])
                operations.append(CommitOperationAdd(path_in_repo=sf_in_repo, path_or_fileobj=sf_filename))
            except Exception as e:
                errors.append((pt_filename, e))
    return operations, errors


def convert(
    api: "HfApi", model_id: str, revision: Optional[str] = None, force: bool = False
) -> Tuple["CommitInfo", List[Tuple[str, "Exception"]]]:
    info = api.model_info(model_id, revision=revision)
    filenames = set(s.rfilename for s in info.siblings)

    # with TemporaryDirectory() as d:
    d = os.environ["HF_HOME"]
    folder = os.path.join(d, "hub", repo_folder_name(repo_id=model_id, repo_type="model"))
    print(f"current folder is {folder}")
    os.makedirs(folder, exist_ok=True)
    library_name = getattr(info, "library_name", None)
    if any(filename.endswith(".safetensors") for filename in filenames) and not force:
        raise AlreadyExists(f"Model {model_id} is already converted, skipping..")
    elif library_name == "transformers":
        discard_names = get_discard_names(model_id, revision=revision, folder=folder, token=api.token)
        if "pytorch_model.bin" in filenames:
            operations, errors = convert_single(
                model_id, revision=revision, folder=folder, token=api.token, discard_names=discard_names
            )
        else:
            raise RuntimeError(f"Model {model_id} doesn't seem to be a valid pytorch model. Cannot convert")
    else:
        operations, errors = convert_generic(
            model_id, revision=revision, folder=folder, filenames=filenames, token=api.token
        )
    return errors


if __name__ == "__main__":
    DESCRIPTION = """
    Simple utility tool to convert automatically some weights on the hub to `safetensors` format.
    It is PyTorch exclusive for now.
    It works by downloading the weights (PT), converting them locally, and uploading them back
    as a PR on the hub.
    """
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "model_id",
        type=str,
        help="The name of the model on the hub to convert. E.g. `gpt2` or `facebook/wav2vec2-base-960h`",
    )
    parser.add_argument(
        "--revision",
        type=str,
        help="The revision to convert",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Create the PR even if it already exists of if the model was already converted.",
    )
    parser.add_argument(
        "-y",
        action="store_true",
        help="Ignore safety prompt",
    )
    args = parser.parse_args()

    model_id = args.model_id
    api = HfApi()
    if args.y:
        txt = "y"
    else:
        txt = input(
            "This conversion script will unpickle a pickled file, which is inherently unsafe. If you do not trust this file, we invite you to use"
            " https://huggingface.co/spaces/safetensors/convert or google colab or other hosted solution to avoid potential issues with this file."
            " Continue [Y/n] ?"
        )
    if txt.lower() in {"", "y"}:
        commit_info, errors = convert(api, model_id, revision=args.revision, force=args.force)
        string = """### Success 🔥 Yay! This model was successfully converted"""
        if errors:
            string += "\nErrors during conversion:\n"
            string += "\n".join(f"Error while converting {filename}: {e}, skipped conversion" for filename, e in errors)
        print(string)
    else:
        print(f"Answer was '{txt}' aborting.")