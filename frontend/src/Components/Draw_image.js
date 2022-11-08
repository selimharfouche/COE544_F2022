import React, { useRef, useState } from "react";
import DeleteIcon from "@mui/icons-material/Delete";
import SendIcon from '@mui/icons-material/Send';
import IconButton from "@mui/material/IconButton";
import CanvasDraw from "react-canvas-draw";
import axios from "axios";
import { Typography, Row } from "antd";

import styles from "./Draw_image.module.css";

function Draw_image() {
  // const [panZoom, setPanZoom] = useState(false);
  // const changeBrush = () => setPanZoom(!panZoom);
  const canvasRef = useRef(CanvasDraw);
  const [axiosResponse, getAxiosResponse] = useState();
  const { Title, Text } = Typography;

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
      }).then(response => {
        getAxiosResponse(response.status);
        if (response.status == 201) {
          window.location = ('/train')
        }
        console.log(response);
      });
    } catch (error) {
      console.log(error)
    }
  }

  return (
    <div className={styles.container}>
      <div style={{ paddingBottom: '150px' }}>
        <Title>Drawn any digit or character and try it with our AI!</Title>
        <Text style={{ paddingLeft: '250px' }}> Feed the AI with your very own drawings</Text>
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
        <Row>
          <div style={{paddingRight:'50px'}}>
            <IconButton
              aria-label="delete"
              onClick={() => canvasRef.current.clear()}
            >
              <DeleteIcon />
            </IconButton>

          </div>

          <div>
            <IconButton
              onClick={handleSubmit}
            >
              <SendIcon />
            </IconButton>
          </div>
        </Row>
      </div>



    </div>
  );
}

export default Draw_image;
