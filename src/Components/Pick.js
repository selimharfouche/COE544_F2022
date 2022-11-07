import React from "react";
import { Button, Typography, Form, Upload } from "antd";


import styles from "./Pick.module.css";
import Pick_u from "./Pick_u";
import Pick_d from "./Pick_d";

const {Text, Title} = Typography;

function Pick() {

  return (
    <>
    <div >

      <div className={styles.container}>
      <Title className={styles["header"]}>How do you want to proceed?</Title>
      <Text className={styles["text"]}>Pick the method you want to test our AI!</Text>
      </div>
    

   <div className={styles["float-container"]}>
        <div className={styles["float-child"]}>    <Pick_u/> </div>
        <div className={styles["float-child"]}>    <Pick_d/> </div>
   </div>

      </div>
    </>
  );
}

export default Pick;
