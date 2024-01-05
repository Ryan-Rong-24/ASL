# Converts mkv files to mp4
for i in *.mkv; do
    ffmpeg -i "$i" -codec copy "${i%.*}.mp4"
done
# Converts webm files to mp4
for i in *.webm; do
    ffmpeg -fflags +genpts -i "$i" -r 24 "${i%.*}.mp4"
done

