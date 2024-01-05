import React, { useEffect, useRef } from 'react';
import './index.css';
import axios from 'axios'

const App = () => {
  const videoRef = useRef(null);
  const photoRef = useRef(null);

  var form = new FormData();
  var count = 0;

  useEffect(() => {
    getVideo();
  }, [videoRef]);

  const send = () => {
    var settings = {
      "url": "https://audiblemotion.azure-api.net/wav3/score",
      "method": "POST",
      "timeout": 0,
      "headers": {
        "Authorization": "Bearer XXX"
      },
      "data": form
    };

    axios(settings).then(response => {
      document.getElementById("prediction").innerHTML = "Sign: "+response.data['sign']
    }).catch(error => {
      console.log(error)
    })
  }

  const getVideo = () => {
    navigator.mediaDevices
      .getUserMedia({ video: { width: 300 } })
      .then((stream) => {
        let video = videoRef.current;
        video.srcObject = stream;
        video.play();
      })
      .catch((err) => {
        console.error('error:', err);
      });
  };

  const paintToCanvas = () => {
    let video = videoRef.current;
    let photo = photoRef.current;
    let ctx = photo.getContext('2d');

    const width = 320;
    const height = 240;
    photo.width = width;
    photo.height = height;

    return setInterval(() => {
      let photo = photoRef.current;
  
      const data = photo.toDataURL('image/jpeg');

      if(count<128){
        var byteString = data.split(',')[1];
        // separate out the mime component
        var mimeString = data.split(',')[0].split(':')[1].split(';')[0];
        // write the bytes of the string to an ArrayBuffer
        var ab = new ArrayBuffer(byteString.length);

        var blob = new Blob([ab], {type: mimeString});
        form.append("image"+count,blob)
      }

      document.getElementById("startstop").innerHTML = "Recording..."

      ctx.drawImage(video, 0, 0, width, height);

      count += 1

      if(count < 10){
        document.getElementById("startstop").innerHTML = "Start!"
      } else if(count > 10 && count < 128){
        document.getElementById("prediction").innerHTML = "Sign"
      }else if(count===128){
        send();
        form = new FormData();
      } else if(count < 180 && count > 128){
        document.getElementById("startstop").innerHTML = "Thinking..."
      } else if(count === 180){
        count = 0
      }

    
    }, 50);
  };

  return (
    <div className="h-screen w-screen m-0 p-0 flex justify-center container">
      <div className="webcam-video h-full w-6/12 flex">
        <div class="grid grid-cols-2 w-full h-fit">
          <img class="col-span-2 mb-3" src="audible-motion-logo.svg"/>
          <div class="col-span-2 w-full h-fit mb-[-2rem]">
            <video
              onCanPlay={() => paintToCanvas()}
              ref={videoRef}
              className="player w-full rounded-md"
            />
            <canvas ref={photoRef} className="photo w-full"  />
          </div>
          <div class="status-wrapper col-span-2 grid grid-cols-2">
            <p id="startstop" class="status bg-[#111111aa] px-6 py-4">Start!</p>
            <p id="prediction" class="status bg-[#000000aa] px-6 py-4">Sign:</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;
