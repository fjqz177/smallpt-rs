for file in *.ppm; do
    ffmpeg -i "$file" "${file:r}.png"
done
