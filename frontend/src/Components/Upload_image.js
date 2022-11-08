import React , { useState } from "react";
import { Button, Typography, Form, Upload } from "antd";
import { InboxOutlined } from "@ant-design/icons";
import axios from "axios";

import styles from "./Upload_image.module.css";

const { Title, Text } = Typography;

function Upload_image() {

  const [selectedFile, setSelectedFile] = React.useState(null);
  const [axiosResponse, getAxiosResponse] = useState();

  const handleSubmit = async event => {
    event.preventDefault()
    const formData = new FormData();
    formData.append("image", selectedFile);
    try {
      const response = await axios({
        method: "post",
        url: "http://127.0.0.1:5000/save-image",
        data: formData,
        headers: { "Content-Type": "multipart/form-data" },
      }).then(response =>{ 
        getAxiosResponse(response.status);
        if (response.status==201) {
          window.location =('/train')
        }
        console.log(response);
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


  const handleFileSelect = (event) => {
    setSelectedFile(event.target.files[0])
  }


  return (
    <>
      <div>
        <div className={styles.container}>
          <Title className={styles["header"]}>Upload Image</Title>
          <Text className={styles["text"]}>
            Select or drop the image you want to test our AI with!
          </Text>
        </div>

        <div style={{paddingLeft:'650px'}}>
          <form onSubmit={handleSubmit}>
            <div style={{paddingBottom:'20px'}}>
            <input type="file" name="ABC" onChange={handleFileSelect} />
            </div>
            <div>
            <input type="submit" value="Upload File" />
            </div>
          </form>
        </div>
      </div>
    </>
  );
}

export default Upload_image;
