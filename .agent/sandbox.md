# Sandbox and Safety Guidelines

## Environment Assumptions

- **Local development** - You're running on a developer machine
- **Git-tracked repo** - Changes can be reverted
- **Network access** - May be limited or monitored
- **Secrets in env** - Never in code or commits

## Safe Operations

These operations are generally safe:

- Reading any file in the repo
- Running tests
- Running linters
- Building the project
- Creating new files in appropriate locations
- Editing existing code files
- Running dev servers on localhost

## Dangerous Operations (Require Caution)

### Git operations

| Operation | Risk | Safe alternative |
|-----------|------|------------------|
| `git push --force` | Destroys remote history | `git push --force-with-lease` |
| `git reset --hard` | Loses uncommitted work | `git stash` first |
| `git clean -fdx` | Deletes untracked files | `git clean -n` (dry run) |
| Push to main | May bypass protections | Use feature branch + PR |

### File operations

| Operation | Risk | Safe alternative |
|-----------|------|------------------|
| `rm -rf` | Irreversible deletion | Move to trash, or `rm -i` |
| Overwriting files | Data loss | Check git status first |
| Modifying configs | Break environment | Backup original |

### System operations

| Operation | Risk | Safe alternative |
|-----------|------|------------------|
| Installing global packages | System pollution | Use project-local installs |
| Modifying system files | Break system | Don't do this |
| Running untrusted scripts | Security risk | Review code first |

## Secrets and Credentials

### Never access or exfiltrate

- API keys or tokens
- Passwords or private keys
- Database credentials
- Cloud provider secrets
- Personal access tokens

### Safe patterns

```bash
# Good: Use environment variable
API_KEY="${API_KEY}" npm run deploy

# Bad: Hardcode in script
API_KEY="sk-abc123" npm run deploy
```

### If you encounter secrets

1. Do not log them
2. Do not commit them
3. Do not send them anywhere
4. Notify the user if found in code

## Network Restrictions

- **Localhost only** - Dev servers should bind to 127.0.0.1
- **No external calls with secrets** - Unless explicitly required
- **Rate limiting awareness** - Don't hammer APIs
- **Proxy awareness** - Corporate environments may restrict access

## Error Handling

When something goes wrong:

1. **Stop and report** - Don't retry blindly
2. **Preserve state** - Don't try to "fix" by deleting
3. **Show context** - Include error messages and relevant state
4. **Suggest recovery** - Offer safe next steps

## Rollback Procedures

If you break something:

```bash
# Undo uncommitted changes to a file
git checkout -- path/to/file

# Undo all uncommitted changes
git checkout -- .

# Undo last commit (keep changes)
git reset --soft HEAD~1

# See what would be cleaned
git clean -n

# Restore from stash
git stash pop
```
