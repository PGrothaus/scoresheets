# remove old builds
docker ps -aq --no-trunc -f status=exited | xargs docker rm
docker images -qf "dangling=true" | xargs docker rmi

# build new docker-images
docker build --rm -t pgn:latest -f buildbits/Dockerfile .
