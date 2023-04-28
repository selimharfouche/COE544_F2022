import React from "react";
import { Button, Typography } from "antd";
import { Link } from 'react-router-dom';

import styles from "./WelcomePage.module.css";

const { Title, Text } = Typography;

function WelcomePage() {
  return (
    <div className={styles.container}>
      <div>
        <Title className={styles["header"]}>Welcome to our project!</Title>
        <Text style={{ color: '#00000073'}}>
          Our project aims to create an intelligent hand-written character
          recognizer agent, to recognize a person’s hand-written characters.
        </Text>
      
      </div>
      <Button type="primary" className={styles.button}>
        <Link to='/pick'>Show Me!</Link>
      </Button> 
      
    </div>
  );
}

export default WelcomePage;
