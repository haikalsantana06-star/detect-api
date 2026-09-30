# GitHub Actions Setup

## Required Secrets

Configure these secrets in your GitHub repository under **Settings → Secrets and variables → Actions**.

### For CI Pipeline (`ci.yml`)

| Secret | Required | Description |
|--------|----------|-------------|
| `CODECOV_TOKEN` | Optional | Token for Codecov coverage reports |

### For Docker Build

| Secret | Required | Description |
|--------|----------|-------------|
| `DOCKER_USERNAME` | Yes | Docker Hub username |
| `DOCKER_TOKEN` | Yes | Docker Hub access token |

### For Deploy Pipeline (`deploy.yml`)

| Secret | Required | Description |
|--------|----------|-------------|
| `RAILWAY_TOKEN` | Yes | Railway API token |

## Getting Tokens

### Railway Token
1. Install Railway CLI: `npm i -g @railway/cli`
2. Login: `railway login`
3. Link project: `railway link`
4. Get token: `railway token`

### Docker Hub Token
1. Go to [Docker Hub Account Settings](https://hub.docker.com/settings/security)
2. Create New Access Token
3. Copy the token value

## Pipeline Triggers

| Pipeline | Trigger | Manual Trigger |
|----------|---------|-----------------|
| `ci.yml` | Push to `main`, PRs to `main` | `docker-build` job via workflow_dispatch |
| `deploy.yml` | Push tags matching `v*` | No |

## Tag a Release

```bash
# Update version in code
git tag v1.0.0
git push origin v1.0.0
```

## Local Railway Deployment

To test locally before CI:

```bash
cd backend
railway login
railway link
railway up
railway status
```
