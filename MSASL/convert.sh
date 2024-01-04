# Converts mkv files to mp4
for i in *.mkv; do
    # ffmpeg -i "$i" -map 0 -c copy -c:a aac "${i%.*}.mp4"
    ffmpeg -i "$i" -codec copy "${i%.*}.mp4"
done
# Converts webm files to mp4
for i in *.webm; do
    ffmpeg -fflags +genpts -i "$i" -r 24 "${i%.*}.mp4"
done

# ffmpeg -i __-0GelCb31wA.mkv -map 0 -c copy -c:a aac __-0GelCb31wA.mp4
# ffmpeg -i __-kj0sejWZlA.mkv -map 0 -c copy -c:a aac __-kj0sejWZlA.mp4
# ffmpeg -i __-QZYGYEZlQg.mkv -map 0 -c copy -c:a aac __-QZYGYEZlQg.mp4
# ffmpeg -i __-tZBgPMjjZM.mkv -map 0 -c copy -c:a aac __-tZBgPMjjZM.mp4

# ffmpeg -fflags +genpts -i "__-ChOGBEL5uE.webm" -r 24 "__-ChOGBEL5uE.mp4"
# ffmpeg -fflags +genpts -i "__-hzoIOWYJac.webm" -r 24 "__-hzoIOWYJac.mp4"
# ffmpeg -fflags +genpts -i "__-msMbTjOPyk.webm" -r 24 "__-msMbTjOPyk.mp4"
# ffmpeg -fflags +genpts -i "__-O61XJs7z1o.webm" -r 24 "__-O61XJs7z1o.mp4"


