$input_json = [Console]::In.ReadToEnd() | ConvertFrom-Json
$prompt = $input_json.prompt

if ($prompt -notmatch '(?i)gitfuck') {
    exit 0
}

$context = @"
Trigger word "gitfuck" detected. Run the full git sync workflow now, for every repo in this workspace that has uncommitted or unpushed changes (including this one):

1. `git status` and `git diff` — inspect changes, note anything unexpected (surprise deletions, unrelated files).
2. `git pull` — pull remote changes first. If this repo has a history-rewrite/force-push situation, use `git fetch origin` + `git reset --hard origin/<branch>` instead per repo conventions, never a merge-pull across rewritten history.
3. If this is the atomic-mnist-problems repo and code under a problem folder changed, run `python test_problem.py` and `python benchmark.py` for that problem before committing.
4. `git add` the relevant changes.
5. Commit with an honest message describing what changed, including any bugs, glitches, or confusing/unfinished state — do not sanitize the message to look cleaner than the actual state of the code.
6. `git push`.
7. After finishing, run `git status` again and confirm the working tree is clean (no untracked or modified files left behind). If it is not clean, keep resolving until it is, or report exactly why it can't be.
"@

@{
    hookSpecificOutput = @{
        hookEventName = "UserPromptSubmit"
        additionalContext = $context
    }
} | ConvertTo-Json -Depth 5
exit 0
