import { Component, OnInit } from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import { math } from '@tensorflow/tfjs';
import * as netUtils from '../neutils';
import { DenseLayerConfig } from '@tensorflow/tfjs-layers/dist/layers/core';
//import * as training from '@te'

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent implements OnInit {

  title = 'tfTest1';

  public test1(): void {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

    model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

    // Generate some synthetic data for training.
    const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
    const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

    // Train the model using the data.
    model.fit(xs, ys, { epochs: 10 }).then(() => {
      // Use the model to do inference on a data point the model hasn't seen before:
      model.predict(tf.tensor2d([5], [1, 1])).toString();
      // Open the browser devtools to see the output
    });
  }

  public setup() {
    // Pass an array of values to create a vector.
    const values = [100];
    for (let i = 0; i < 30; i++) {
      values[i] = Math.random() * 100;
    }
    const shape: [number, number, number] = [2, 5, 3];
    const tense = tf.tensor3d(values, shape, 'int32');
    console.log(tense.data());
    /* tense.data(function(stuff){
      console.log(stuff);
    }); */
    tense.data().then(function (stuff) {
      console.log(stuff);
    });
  }

  public testMult() {
    const a = tf.tensor2d([1, 2], [1, 2]);
    const b = tf.tensor2d([1, 2, 3, 4], [2, 2]);
    const c = a.matMul(b);
    console.log(c.toString());
  }

  public testMemory() {
    function repeater() {
      const mm= tf.memory();
      console.log(`cicle ${count} tensors: ${mm.numTensors} mem: ${mm.numBytes}`);
      const values = [15000];
      for (let i = 0; i < 150000; i++) {
        values[i] = Math.random() * 100;
      }
      const shape:[number, number, number]=[1, 500, 300];
      const a= tf.tensor3d(values, shape, 'int32');
      const b= tf.tensor3d(values, shape, 'int32');
      const bb= b.transpose();
      const c= a.mul(bb);
      a.dispose();
      b.dispose();
      bb.dispose();
      c.dispose();

    };
    let count = 0;
    netUtils.smplUntil(
      () => { return (count++ > 50); },
      repeater,
      (err)=>console.error(err)
    );

  }

  public testModel1(){
    const model= tf.sequential();

    const hidden= tf.layers.dense({
      units: 4,
      inputShape: [2],
      activation: 'sigmoid'
    });
    const output = tf.layers.dense({
      units: 3,
      activation: 'sigmoid'
    });
    model.add(hidden);
    model.add(output);
    const sdgOpt= tf.train.sgd(0.1);
    const modelCfg = {
      optimizer: sdgOpt,
      loss: tf.losses.meanSquaredError
    };
    model.compile(modelCfg);

  }

  ngOnInit(): void {
    console.log("ngOnInit");
  }
}
