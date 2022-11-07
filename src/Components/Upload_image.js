import React from "react";
import { Button, Typography, Form, Upload } from "antd";
import { InboxOutlined } from "@ant-design/icons";
import axios from "axios";

import styles from "./Upload_image.module.css";

const { Title, Text } = Typography;

function Upload_image() {

  const [selectedFile, setSelectedFile] = React.useState(null);

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
      });
    } catch(error) {
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

        <div>
          <form onSubmit={handleSubmit}>
            <input type="file" name="ABC" onChange={handleFileSelect} />
            <input type="submit" value="Upload File" />
          </form>
        </div>
      </div>
    </>
  );
}

export default Upload_image;
