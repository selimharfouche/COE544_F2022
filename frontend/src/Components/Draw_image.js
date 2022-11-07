import React, { useRef } from "react";
import DeleteIcon from "@mui/icons-material/Delete";
import SendIcon from '@mui/icons-material/Send';
import IconButton from "@mui/material/IconButton";
import CanvasDraw from "react-canvas-draw";
import axios from "axios";

import "./Draw_image.module.css";

function Draw_image() {
  // const [panZoom, setPanZoom] = useState(false);
  // const changeBrush = () => setPanZoom(!panZoom);
  const canvasRef = useRef(CanvasDraw);

  const handleSubmit = async event => {
    const image64 = canvasRef.current.getDataURL("image/jpeg", null, '#FFF');
    const formData = new FormData();
    formData.append("drawnImage", image64)
    try {
      const response = await axios({
        method: "post",
        url: "http://127.0.0.1:5000/save-drawn-image",
        data: formData,
        headers: { "Content-Type": "multipart/form-data" },
      });
    } catch(error) {
      console.log(error)
    }
  }

  return (
    <div>
      <div>
        <h1>Hello from draw</h1>
      </div>

      <div>
        <CanvasDraw
          ref={canvasRef}
          className="drawer"
          backgroundColor="#fff"
          brushColor="#000"
          hideGrid={true}
        />
      </div>

      <div>
        <IconButton
          aria-label="delete"
          onClick={() => canvasRef.current.clear()}
        >
          <DeleteIcon />
        </IconButton>

        <IconButton
        onClick={handleSubmit}
        >
          <SendIcon />
        </IconButton>
      </div>

    
    </div>
  );
}

export default Draw_image;
