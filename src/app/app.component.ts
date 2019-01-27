import { Component, OnInit } from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import { math, Tensor } from '@tensorflow/tfjs';
import * as netUtils from '../neutils';
import { DenseLayerConfig } from '@tensorflow/tfjs-layers/dist/layers/core';
// import * as training from '@te'
import * as p5 from 'p5';
import 'p5/lib/addons/p5.sound';
import 'p5/lib/addons/p5.dom';
import { callAndCheck } from '@tensorflow/tfjs-core/dist/kernels/webgl/webgl_util';



@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent implements OnInit {

  title = 'tfTest1';
  private canvasP5: p5;
  private x_vals = [];
  private y_vals = [];
  private m: Tensor ;
  private b: Tensor ;
  private learningRate = 0.2;
  optimizer = tf.train.sgd(this.learningRate);

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

    }
    let count = 0;
    netUtils.smplUntil(
      () => (count++ > 50),
      repeater,
      (err)=>console.error(err)
    );

  }

  public testModel1(){
    const model= tf.sequential();
    // Dense is a fully conected (todos con todos entre capas)
    const hidden = tf.layers.dense({
      units: 4,  // Number of nodes
      inputShape: [2],   // number of input shape
      activation: 'sigmoid'
    });
    const output = tf.layers.dense({
      units: 3,
      // El input shape se saca del anterior automaticamente
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

    const inputData= tf.tensor2d([[0.25, 0.9]]);
    const outputData= model.predict(inputData);
    console.log(outputData.toString());

  }

  public testTrainModel1(){
    const model = tf.sequential();
    // Dense is a fully conected (todos con todos entre capas)
    const hidden = tf.layers.dense({
      units: 4,  // Number of nodes
      inputShape: [2],   // number of input shape
      activation: 'sigmoid'
    });
    const output = tf.layers.dense({
      units: 3,
      // El input shape se saca del anterior automaticamente
      activation: 'sigmoid'
    });
    model.add(hidden);
    model.add(output);
    const sdgOpt = tf.train.sgd(0.1);
    const modelCfg = {
      optimizer: sdgOpt,
      loss: tf.losses.meanSquaredError
    };
    model.compile(modelCfg);

    const xs = tf.tensor2d([
      [0.35, 0.92],
      [0.12, 0.3],
      [0.4, 0.74]
    ]);
    const ys = tf.tensor2d([
      [0.1, 0.1, 0.02],
      [0.4, 0.01, 0.22],
      [0.2, 0.9, 0.02]
    ]);
    model.fit(xs, ys, {verbose: 1, epochs: 5}).then( (history) => console.log(history));

    // const inputData= tf.tensor2d([[0.25, 0.9]]);
    // const outputData= model.predict(inputData);
    // console.log(outputData.toString());

  }

  ngOnInit(): void {
    console.log('ngOnInit');
  }

  public setupLinearRegretion() {


    const sketch = (s) => {
      s.preload = () => {
        // preload code
      };
      s.setup = () => {
        const cc = s.createCanvas(400, 400);
        //cc.parent('myContainer');
      };
      // s.draw = () => {
        // s.background(255);
        // s.rect(100, 100, 100, 100);
      // };
    };
    this.canvasP5 = new p5(sketch);
    this.canvasP5.mousePressed = (event?: object) => {
      this.mousePressed(event);
    };
    // m y b son las variuablñes a buscar, y pueden cambiar
    this.m = tf.variable(tf.scalar(this.canvasP5.random(1)));
    this.b = tf.variable(tf.scalar(this.canvasP5.random(1)));

    this.canvasP5.draw = () => {
      this.draw();
    };


  }

  loss(pred, labels) {
    // Realiza la resta de lo obtenido, menos lo predicho al cuadrado y la media de eso
    return pred.sub(labels).square().mean();
  }


  /**
   * Pasandole los valor de x, devuelve los calculados y para la recta
   *
   * @param {*} x
   * @returns {Tensor} - Un array con los valores correspondientes para cada entrada
   * @memberof AppComponent
   */
  predict(x: Array<number>): Tensor {
    const xs = tf.tensor1d(x);  // se pasa de un array normal a un tensor
    // y = mx + b; Ys son los valores teoricos de la recta en los muntos obteniudos
    const ys = xs.mul(this.m).add(this.b);
    return ys;
  }

  mousePressed(event?: object): void {
    console.log('mouse pressed');
    // const x= map(mouseX,0, width, 0, 1);
    // Se mapean cordenadas de canvas a 0, 1
    const x = this.canvasP5.map(this.canvasP5.mouseX, 0, this.canvasP5.width, 0, 1);
    const y = this.canvasP5.map(this.canvasP5.mouseY, 0, this.canvasP5.height, 1, 0);
    this.x_vals.push(x);
    this.y_vals.push(y);
  }

  drawt(){
    console.log('d');
  }

  /** Que se realiza en el bucle de pintado */
  draw(): void {
    tf.tidy(() => {  // Se liberan automaticamente los tensores creados dentro
      if (this.x_vals.length > 0) {
        const ys2 = tf.tensor1d(this.y_vals);
        const yCalculated = this.predict(this.x_vals);
        this.optimizer.minimize(() => this.loss(yCalculated, ys2));
      }
    });

    this.canvasP5.background(0);

    this.canvasP5.stroke(255);
    this.canvasP5.strokeWeight(8);
    for (let i = 0; i < this.x_vals.length; i++) {  //Se pintan los puntos pulsados
      const px = this.canvasP5.map(this.x_vals[i], 0, 1, 0, this.canvasP5.width);
      const py = this.canvasP5.map(this.y_vals[i], 0, 1, this.canvasP5.height, 0);
      this.canvasP5.point(px, py);
    }


    const lineX = [0, 1];

    const ys = tf.tidy(() => {
      return this.predict(lineX);
    });
    const lineY = ys.dataSync();
    ys.dispose();

    const x1 = this.canvasP5.map(lineX[0], 0, 1, 0, this.canvasP5.width);
    const x2 = this.canvasP5.map(lineX[1], 0, 1, 0, this.canvasP5.width);

    const y1 = this.canvasP5.map(lineY[0], 0, 1, this.canvasP5.height, 0);
    const y2 = this.canvasP5.map(lineY[1], 0, 1, this.canvasP5.height, 0);

    this.canvasP5.strokeWeight(2);
    this.canvasP5.line(x1, y1, x2, y2);

    console.log(tf.memory().numTensors);
    // noLoop();
  }

}
