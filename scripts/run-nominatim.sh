#!/usr/bin/sh

docker run -it --rm --shm-size=1g \
  -e PBF_URL=https://download.geofabrik.de/europe/italy/centro-latest.osm.pbf \
  -e REPLICATION_URL=https://download.geofabrik.de/europe/italy-updates/ \
  -e IMPORT_WIKIPEDIA=true \
  -e NOMINATIM_PASSWORD=very_secure_password \
  -v nominatim-data:/var/lib/postgresql/12/main \
  -p 8080:8080 \
  --name nominatim \
  mediagis/nominatim:4.0
