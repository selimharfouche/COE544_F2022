import React from "react";
import { Button, Typography, Form, Upload } from "antd";
import { FormatPainterOutlined } from "@ant-design/icons";

import styles from "./Pick.module.css";
import { Link } from "react-router-dom";

const { Text } = Typography;

function Pick_d() {
  return (
    <div>
      <div className={styles.container}>
        <div>
        <FormatPainterOutlined style={{fontSize: '80px', paddingBottom: 20}}/>
        </div>
        <div>
          <Text>Draw an image</Text>
        </div>
        <div>
          <Button type="primary">
            <Link to="/draw"> Click here! </Link>
          </Button>
        </div>
      </div>
    </div>
  );
}

export default Pick_d;
