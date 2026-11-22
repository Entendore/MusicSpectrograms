#!/bin/bash

if [ -z "$1" ]; then
    read -p "Enter YouTube URL: " url
else
    url="$1"
fi

echo "Downloading audio from $url..."

# -x: Extract audio
# --audio-format mp3: Convert to mp3
# --audio-quality 0: Best quality
yt-dlp -x --audio-format mp3 --audio-quality 0 -o "%(title)s.%(ext)s" "$url"

echo "Download finished."