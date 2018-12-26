import { Component, OnInit } from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import { math } from '@tensorflow/tfjs';

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
      model.predict(tf.tensor2d([5], [1, 1])).print();
      // Open the browser devtools to see the output
    });
  }

  public setup() {
    // Pass an array of values to create a vector.
    const values=[100];
    for(let i=0; i< 30; i++){
      values[i]= Math.random()*100;
    }
    const shape: [number, number, number]=[2,5,3];
    const data=tf.tensor3d(values, shape, 'int32');
    console.log(data.toString());
  }

  ngOnInit(): void {
    console.log("ngOnInit");
  }
}
