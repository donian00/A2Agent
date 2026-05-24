import logging
import os
import subprocess
logger = logging.getLogger(__name__)

def get_repo_dir_name(repo: str):
    return repo.replace("/", "_")


def setup_github_repo(repo: str, base_commit: str = None, base_dir: str = "/tmp/repos") -> str:
    repo_name = get_repo_dir_name(repo)
    repo_url = f"https://github.com/{repo}.git"
    path = f"{base_dir}/{repo_name}"
    logger.info(
        f"Clone Github repo {repo_url} to {path} and checkout commit {base_commit}"
    )
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"Directory '{path}' was created.")

    # Try cache-aware clone: reuse a shared cache to avoid repeated network clones
    # Also check smith_repos (double-underscore naming convention) — but only if it's
    # NOT a shallow clone (shallow caches lack older commits required by SWE-Gym).
    smith_repo_name = repo.replace("/", "__")
    smith_cache = os.path.join("playground", "smith_repos", smith_repo_name)
    cache_dir = os.path.join("playground", "repo_cache", repo_name)
    use_smith = False
    if os.path.exists(f"{smith_cache}/.git") and not os.path.exists(f"{cache_dir}/.git"):
        try:
            shallow = subprocess.run(
                ["git", "rev-parse", "--is-shallow-repository"],
                cwd=smith_cache, check=True, text=True, capture_output=True,
            ).stdout.strip()
            if shallow == "false":
                use_smith = True
        except subprocess.CalledProcessError:
            pass
    if use_smith:
        cache_dir = smith_cache
    if os.path.exists(f"{cache_dir}/.git") and not os.path.exists(f"{path}/.git"):
        logger.info(f"Local clone from cache: {cache_dir} -> {path}")
        try:
            subprocess.run(
                ["git", "clone", "--local", cache_dir, path],
                check=True, text=True, capture_output=True,
            )
        except subprocess.CalledProcessError:
            logger.warning("Local clone from cache failed, falling back to network clone")
            maybe_clone(repo_url, path)
    else:
        maybe_clone(repo_url, path)

    if base_commit:
        checkout_commit(path, base_commit)

    # Populate cache if this was the first network clone for this repo
    if not os.path.exists(f"{cache_dir}/.git") and os.path.exists(f"{path}/.git"):
        try:
            os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
            logger.info(f"Populating repo cache: {path} -> {cache_dir}")
            subprocess.run(
                ["git", "clone", "--local", "--no-checkout", path, cache_dir],
                check=True, text=True, capture_output=True,
            )
        except Exception as e:
            logger.warning(f"Failed to populate repo cache: {e}")

    return path


def maybe_clone(repo_url, repo_dir):
    if not os.path.exists(f"{repo_dir}/.git"):
        logger.info(f"Cloning repo '{repo_url}'")
        # Clone the repo if the directory doesn't exist
        result = subprocess.run(
            ["git", "clone", repo_url, repo_dir],
            check=True,
            text=True,
            capture_output=True,
        )

        if result.returncode == 0:
            logger.info(f"Repo '{repo_url}' was cloned to '{repo_dir}'")
        else:
            logger.info(f"Failed to clone repo '{repo_url}' to '{repo_dir}'")
            raise ValueError(f"Failed to clone repo '{repo_url}' to '{repo_dir}'")


def pull_latest(repo_dir):
    subprocess.run(
        ["git", "pull"],
        cwd=repo_dir,
        check=True,
        text=True,
        capture_output=True,
    )


def clean_and_reset_state(repo_dir):
    subprocess.run(
        ["git", "clean", "-fd"],
        cwd=repo_dir,
        check=True,
        text=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "reset", "--hard"],
        cwd=repo_dir,
        check=True,
        text=True,
        capture_output=True,
    )


def create_branch(repo_dir, branch_name):
    try:
        subprocess.run(
            ["git", "branch", branch_name],
            cwd=repo_dir,
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error(e.stderr)
        raise e


def create_and_checkout_branch(repo_dir, branch_name):
    try:
        branches = subprocess.run(
            ["git", "branch"],
            cwd=repo_dir,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.split("\n")
        branches = [branch.strip() for branch in branches]
        if branch_name in branches:
            subprocess.run(
                ["git", "checkout", branch_name],
                cwd=repo_dir,
                check=True,
                text=True,
                capture_output=True,
            )
        else:
            subprocess.run(
                ["git", "checkout", "-b", branch_name],
                cwd=repo_dir,
                check=True,
                text=True,
                capture_output=True,
            )
    except subprocess.CalledProcessError as e:
        logger.error(e.stderr)
        raise e


def commit_changes(repo_dir, commit_message):
    subprocess.run(
        ["git", "commit", "-m", commit_message, "--no-verify"],
        cwd=repo_dir,
        check=True,
        text=True,
        capture_output=True,
    )


def checkout_branch(repo_dir, branch_name):
    subprocess.run(
        ["git", "checkout", branch_name],
        cwd=repo_dir,
        check=True,
        text=True,
        capture_output=True,
    )


def push_branch(repo_dir, branch_name):
    subprocess.run(
        ["git", "push", "origin", branch_name, "--no-verify"],
        cwd=repo_dir,
        check=True,
        text=True,
        capture_output=True,
    )


def get_diff(repo_dir):
    output = subprocess.run(
        ["git", "diff"], cwd=repo_dir, check=True, text=True, capture_output=True
    )

    return output.stdout


def stage_all_files(repo_dir):
    subprocess.run(
        ["git", "add", "."], cwd=repo_dir, check=True, text=True, capture_output=True
    )


def checkout_commit(repo_dir, commit_hash):
    try:
        subprocess.run(
            ["git", "reset", "--hard", commit_hash],
            cwd=repo_dir,
            check=True,
            text=True,
            capture_output=True,
        )
        return
    except subprocess.CalledProcessError:
        pass

    # Try SHA fetch (works on GitHub when allowAnySHA1InWant is enabled)
    try:
        subprocess.run(
            ["git", "fetch", "--depth=1", "origin", commit_hash],
            cwd=repo_dir, check=True, text=True, capture_output=True,
        )
        subprocess.run(
            ["git", "reset", "--hard", "FETCH_HEAD"],
            cwd=repo_dir, check=True, text=True, capture_output=True,
        )
        return
    except subprocess.CalledProcessError:
        pass

    # Convert shallow → full, then fetch all branches/tags. Catches the common
    # "commit lives on a branch we didn't pull" case.
    try:
        subprocess.run(
            ["git", "fetch", "--unshallow", "origin"],
            cwd=repo_dir, text=True, capture_output=True,
        )
    except subprocess.CalledProcessError:
        pass
    try:
        subprocess.run(
            ["git", "fetch", "origin", "+refs/heads/*:refs/remotes/origin/*", "--tags"],
            cwd=repo_dir, check=True, text=True, capture_output=True,
        )
        subprocess.run(
            ["git", "reset", "--hard", commit_hash],
            cwd=repo_dir, check=True, text=True, capture_output=True,
        )
        return
    except subprocess.CalledProcessError:
        pass

    # Last resort: keep current HEAD (will likely produce wrong line numbers)
    logger.warning(f"Cannot checkout {commit_hash} in {repo_dir}, using current HEAD")
    subprocess.run(
        ["git", "reset", "--hard", "HEAD"],
        cwd=repo_dir, text=True, capture_output=True,
    )


def create_and_checkout_new_branch(repo_dir: str, branch_name: str):
    try:
        subprocess.run(
            ["git", "checkout", "-b", branch_name],
            cwd=repo_dir,
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error(e.stderr)
        raise e


def setup_repo(repo_url, repo_dir, branch_name="master"):
    maybe_clone(repo_url, repo_dir)
    clean_and_reset_state(repo_dir)
    checkout_branch(repo_dir, branch_name)
    pull_latest(repo_dir)


def clean_and_reset_repo(repo_dir, branch_name="master"):
    clean_and_reset_state(repo_dir)
    checkout_branch(repo_dir, branch_name)
    pull_latest(repo_dir)