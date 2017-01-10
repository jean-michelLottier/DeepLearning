/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.jml.main;

import com.jml.dl.DeepLearning;
import com.jml.nn.Sample;
import com.jml.nn.NeuralNetwork;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

/**
 *
 * @author gt902715 <jean-michel.lottier@alten.com>
 */
public class Main {

	public static final Logger LOGGER = Logger.getLogger(Main.class.getName());
	public static final String IRIS_SETOSA = "Iris-setosa";
	public static final String IRIS_VERSICOLOR = "Iris-versicolor";
	public static final String IRIS_VIRGINICA = "Iris-virginica";

	/**
	 * @param args the command line arguments
	 */
	public static void main(String[] args) throws Exception {
		testIris();
//		NeuralNetwork network = new NeuralNetwork();
//		network.init(3, 0, 1, true, 0.5f);
//		network.feedForward(new Sample(Arrays.asList(0.1f, 0.2f, 0.3f)));
//		for(Node n : network.getContextLayer()) {
//			System.out.println("Node id: " + n.id + " output: " + n.getOutput());
//			for (int key : n.getOutputNodeId()) {
//				System.out.println(n.id + " --> " + key);
//			}
//		}
//		System.out.println("------------------------------");
//		for (Node n : network.getHiddenLayer()) {
//			System.out.println("Node id: " + n.id + " output: " + n.getOutput());
////			for (int key : n.getInputWeights().keySet()) {
////				System.out.println(key + " --> " + n.getInputWeights().get(key));
////			}
//			for (int key : n.getOutputNodeId()) {
//				System.out.println(n.id + " --> " + key);
//			}
//		}

//		for (Node n : network.getInputLayer()) {
//			System.out.println("Node id: " + n.id + " output: " + n.getOutput());
//			for (int key : n.getOutputNodeId()) {
//				System.out.println(n.id + " --> " + key);
//			}
//		}
//
//		for (Node n : network.getOutputLayer()) {
//			System.out.println("Node id: " + n.id + " output: " + n.getOutput());
//			for (int key : n.getInputWeights().keySet()) {
//				System.out.println(key + " --> " + n.getInputWeights().get(key));
//			}
//		}
	}

	public static void testIris() throws Exception {
		File file = new File("iris_dataset.csv");
		String content = readFile(file);
		List<String> contentLines = new ArrayList<>(Arrays.asList(content.split(System.lineSeparator())));
		contentLines.remove(0);
		List<Sample> samples = new ArrayList<>();
		for (String line : contentLines) {
			List<Float> inputs = new ArrayList<>();
			List<Integer> desiredOutputs = new ArrayList<>();
			String[] values = line.split(";");
			desiredOutputs.add(irisClassChoice(values[values.length - 1]));
			for (int i = 0; i < values.length - 1; ++i) {
				inputs.add(Float.valueOf(values[i]));
			}
			samples.add(new Sample(inputs, desiredOutputs));
		}

		Map<String, NeuralNetwork> neuralNetworks = new HashMap<>();
		for (int i = 1; i <= 3; ++i) {
			NeuralNetwork neuralNetwork = new NeuralNetwork();
			neuralNetwork.init(4, 5, 1, false, 0.3f);
			List<Sample> samplesCloned = samples.stream().map(sample -> new Sample<>(sample)).collect(Collectors.toList());
			for (Sample sample : samplesCloned) {
				if ((int) sample.getDesiredOutputs().get(0) == i) {
					sample.getDesiredOutputs().add(0, 1);
				} else {
					sample.getDesiredOutputs().add(0, 0);
				}
			}
//                        System.out.println("++++++++++++ " + i + "(2 = Versicolor) +++++++++++++");
//			neuralNetwork.kFoldCrossValidation(5, samplesCloned, true);
                        neuralNetwork = DeepLearning.calculateBestNeuralNetworks(samplesCloned, false);
                        System.out.println("hidden nodes = " + neuralNetwork.getHiddenLayer().size() + ", learning rate = " + neuralNetwork.getHiddenLayer().get(0).getLearningRate());

			switch (i) {
				case 1:
					neuralNetworks.put(IRIS_SETOSA, neuralNetwork);
				case 2:
					neuralNetworks.put(IRIS_VERSICOLOR, neuralNetwork);
				default:
					neuralNetworks.put(IRIS_VIRGINICA, neuralNetwork);
			}
		}

		XYSeries irisSetosaSeries = new XYSeries(IRIS_SETOSA);
		XYSeries irisVersicolorSeries = new XYSeries(IRIS_VERSICOLOR);
		XYSeries irisVirginicaSeries = new XYSeries(IRIS_VIRGINICA);
		for (Sample sample : samples) {
			Tuple<Integer, Float> finalResult = null;
//			System.out.println("------ " + sample.getValues() + " ------");
			for (String key : neuralNetworks.keySet()) {
				NeuralNetwork nn = neuralNetworks.get(key);
				nn.feedForward(sample);
//				System.out.println("class: " + key + ", output: " + nn.getOutputLayer().get(0).getOutput());
				if (finalResult == null) {
					finalResult = new Tuple(irisClassChoice(key), nn.getOutputLayer().get(0).getOutput());
				} else if (nn.getOutputLayer().get(0).getOutput() > finalResult.getY()) {
					finalResult.setX(irisClassChoice(key));
					finalResult.setY(nn.getOutputLayer().get(0).getOutput());
				}
			}

			double x = Double.valueOf(String.valueOf(sample.getValues().get(2)));
			double y = Double.valueOf(String.valueOf(sample.getValues().get(3)));
			switch (finalResult.getX()) {
				case 1:
					irisSetosaSeries.add(x, y);
					break;
				case 2:
					irisVersicolorSeries.add(x, y);
					break;
				case 3:
					irisVirginicaSeries.add(x, y);
					break;
			}
		}

		System.out.println("irisSetosaSeries size: " + irisSetosaSeries.getItemCount());
		System.out.println("irisVersicolorSeries size: " + irisVersicolorSeries.getItemCount());
		System.out.println("irisVirginicaSeries size: " + irisVirginicaSeries.getItemCount());

		XYSeriesCollection collection = new XYSeriesCollection();
		collection.addSeries(irisSetosaSeries);
		collection.addSeries(irisVersicolorSeries);
		collection.addSeries(irisVirginicaSeries);
		Graph graph = new Graph();
		graph.setDataset(collection);
		graph.drawScatterPlot("Iris result", collection);
	}

	public static String readFile(File file) {
		StringBuilder sb = new StringBuilder();
		try {
			BufferedReader br = new BufferedReader(new FileReader(file));
			String line = br.readLine();
			while (line != null) {
				sb.append(line).append(System.lineSeparator());
				line = br.readLine();
			}

			br.close();
		} catch (FileNotFoundException ex) {
			Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
		} catch (IOException ex) {
			Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
		}

		return sb.toString();
	}

	public static int irisClassChoice(String irisClassName) {
		switch (irisClassName) {
			case IRIS_SETOSA:
				return 1;
			case IRIS_VERSICOLOR:
				return 2;
			default:
				return 3;
		}
	}

	public static class Tuple<K, V> {

		private K x;
		private V y;

		public Tuple(K x, V y) {
			this.x = x;
			this.y = y;
		}

		public K getX() {
			return x;
		}

		public void setX(K x) {
			this.x = x;
		}

		public V getY() {
			return y;
		}

		public void setY(V y) {
			this.y = y;
		}
	}

}
