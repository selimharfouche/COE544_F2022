import React, {useState } from "react";
import { Row, Typography, Form, Radio, Button } from "antd";
import Select from "react-select";
import axios from "axios";
import Swal from 'sweetalert2'


import styles from "./Train.module.css"

const {Text, Title} = Typography;


function Train() {


    const Swal = require('sweetalert2')

    const [trainingAlgo, setTrainingAlgo] = useState('');
    const [features, getFeatures] = useState();
    const [axiosResponse, getAxiosResponse] = useState();

    
    const multiHandle = (event) =>
    {
        getFeatures(Array.isArray(event)?event.map(x => x.label):[]);
    }

    const onFinish = async event => {

        console.log('onFinish ran')
        console.log('trainingAlgo: ', trainingAlgo)
        console.log('features: ', features)

        // event.preventDefault();
        const formData = new FormData();
        formData.append("trainingAlgo", trainingAlgo);

        //add if statent ex: if feature = Pixel Intensity, feature becomes pixel_intensity
        formData.append("features", features);
        try{
             const response = await axios({
                 method:"post",
                 url: "http://127.0.0.1:5000/get-label",
                 data: formData,
                headers: { "Content-Type": "multipart/form-data" },
             }).then(response =>{
                getAxiosResponse(response.data);
                var obtainedData = JSON.stringify(response.data)

                
                Swal.fire({
                    title: obtainedData.substring(9,10),
                    text: 'with confidence: '+obtainedData.substring(14,obtainedData.length-2),
                    icon: 'error',
                    confirmButtonText: 'Cool'
                  })
  
                console.log(response);
             });
            }catch(error){
                console.log(error)
            }
         }
        
    

    var featureName=[
 
        {
            value:1,
            label:"aspect_ratio"
            
        },
        {
            value:2,
            label:"top_half_img"
        },
        {
            value:3,
            label:"lower_half_img"
        },
        {
            value:4,
            label:"right_half_img"
        },
        {
            value:5,
            label:"left_half_img"
        },
        {
            value:6,
            label:"histogram"
        },
        {
            value:7,
            label:"pixel_intensity"
        },
        {
            value:8,
            label:"sobel_edge"
        },
        {
            value:9,
            label:"canny_edge"
        },
        {
            value:10,
            label:"LocalBinaryPatterns"
        },
        {
            value:11,
            label:"HOG"
        },




    ]
    
  


  return (
    <div>

      <div className={styles.container}>
        <Title className={styles["header"]}>How do you want to train your AI?</Title>
        <Text className={styles["text"]}>
          Select the training algorithms and training features to make your own AI!
        </Text>
      </div>

      <div>
        <Form
            name="train_data"
            // className={styles}  ##### For styling later
            onFinish={onFinish}
            >
                <Form.Item 
                    name="trainingAlgo" 
                    label="Training Algorithms: " 
                    onChange={event =>setTrainingAlgo(event.target.value)} 
                    value={trainingAlgo}
                    >
                    <Radio.Group>
                        <Row><Radio value="SVM">SVM</Radio></Row>
                        <Row><Radio value="KNN">KNN</Radio></Row>
                        <Row><Radio value="ensemble">Ensemble</Radio></Row>
                        <Row><Radio value="ensemble">PreTrained</Radio></Row>
                        
                    </Radio.Group>
                </Form.Item>

                <Form.Item 
                    name="features" 
                    label="Select one or more feature(s)"
                    // onChange={event => setFeatures(event.target.value)}

                    // value = {features}
                >
                    <div style={{width:'700px'}}>
                    <Select 
                        isMulti
                        options={featureName} 
                        onChange={multiHandle}                       
                        >
                    </Select>
                    </div>
                </Form.Item>

                <Form.Item wrapperCol={{ span: 12, offset: 6 }}>
                    <Button type="primary" htmlType="submit" value="Upload list">
                        Submit
                    </Button>
                </Form.Item>



        </Form>

        <div>{axiosResponse}</div>

      </div>




    </div>
  );

}

export default Train;
