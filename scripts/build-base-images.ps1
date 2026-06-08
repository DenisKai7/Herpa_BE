$env:DOCKER_BUILDKIT = "1"
$env:COMPOSE_DOCKER_CLI_BUILD = "1"

docker build `
  -f Dockerfile.backend-base `
  -t medical-backend-base:py311-v1 .

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

docker build `
  -f Dockerfile.ocr-base `
  -t medical-ocr-base:torch-cpu-v1 .

exit $LASTEXITCODE
