import React from "react";
import { Button, Typography, Form, Upload } from "antd";
import { UploadOutlined } from "@ant-design/icons";

import styles from "./Pick.module.css";
import { Link } from "react-router-dom";

const { Text } = Typography;
function Pick_u() {
  return (
    <div>
      <div className={styles.container}>
        <div>
        <UploadOutlined style={{fontSize: '80px', paddingBottom: 20}}/>
        </div>
        <div>
          <Text>Upload an image</Text>
        </div>
        <div>
          <Button type="primary" className={styles.button}>
            <Link to="/upload"> Click here! </Link>
          </Button>
        </div>
      </div>
    </div>
  );
}

export default Pick_u;
