import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import axios from 'axios';

import WelcomePage from './Components/WelcomePage';
import Pick from './Components/Pick';
import Upload_image from './Components/Upload_image';
import Draw_image from './Components/Draw_image';
import Train from './Components/Train';
import './App.css';

function App() {
  return (
    <Router>
    <div className="App-background">
    <Routes>
      <Route exact path='/' element={<WelcomePage/>}/>
      <Route path='/pick' element={<Pick/>} />
      <Route path='/upload' element={<Upload_image/>}/>
      <Route path='/draw' element={<Draw_image/>}/>
      <Route path='/train' element={<Train/>}/>

    </Routes>
    </div>
    </Router>
  );
}

export default App;
