/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.jml.dl;

import com.jml.nn.NeuralNetwork;
import com.jml.nn.Sample;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;

/**
 *
 * @author jean-michel
 */
public class DeepLearning {

    private static final float DEFAULT_LEARNING_RATE = 0.3f;
    private static final int DEFAULT_K = 5;

    private static List<NeuralNetwork> neuralNetworks;
    private static int nbInputNodes = 0;
    private static int nbOutputNodes = 0;
    private static int nbHiddenNodes = 0;

    private static final Lock lock = new ReentrantLock();

    /**
     * <p>
     * Algo pour trouver le reseau neuronal ayant la meilleure réussite</p>
     *
     * @param samples
     * @param isRNN
     * @return
     * @throws java.lang.Exception
     */
    public static NeuralNetwork calculateBestNeuralNetworks(List<Sample> samples, boolean isRNN) throws Exception {
        if (samples == null || samples.isEmpty()) {
            return null;
        }

        nbInputNodes = samples.get(0).getValues().size();
        nbOutputNodes = samples.get(0).getDesiredOutputs().size();
        nbHiddenNodes = Math.round((nbInputNodes + nbOutputNodes) / 2) - 1;
        if (nbHiddenNodes <= 0) {
            nbHiddenNodes = nbInputNodes;
        }

        NeuralNetwork bestNn = null;

        neuralNetworks = new ArrayList<>();
        Thread[] threads = new Thread[(nbInputNodes * 2 - nbHiddenNodes)];
        int offset = nbHiddenNodes;
        for (int i = offset; i < (nbInputNodes * 2); i++) {
            threads[i - offset] = new ThreadNN(samples, isRNN);
            threads[i - offset].start();
            nbHiddenNodes++;
        }

        for (Thread thread : threads) {
            thread.join();
        }

        for (NeuralNetwork neuralNetwork : neuralNetworks) {
            if (bestNn == null || neuralNetwork.getCvError() < bestNn.getCvError()) {
                bestNn = neuralNetwork;
            }
        }

        return bestNn;
    }

    private static class ThreadNN extends Thread {

        private final List<Sample> samples;
        private final boolean isRNN;

        public ThreadNN(List<Sample> samples, boolean isRNN) {
            this.samples = samples;
            this.isRNN = isRNN;
        }

        @Override
        public void run() {
            boolean firstTraining = true;
            boolean isRunning = true;
            float learningRate = DEFAULT_LEARNING_RATE;
            int k = DEFAULT_K;
            int _nbInputNodes = nbInputNodes;
            int _nbOutputNodes = nbOutputNodes;
            int _nbHiddenNodes = nbHiddenNodes;

            do {
                if (!firstTraining) {
                    if (learningRate < 0.8f) {
                        learningRate += 0.1f;
                    } else if (k < 10) {
                        learningRate = DEFAULT_LEARNING_RATE;
                        k += 5;
                    }
                } else {
                    firstTraining = false;
                }

                List<Sample> samplesCloned = this.samples.stream().map(sample -> new Sample<>(sample)).collect(Collectors.toList());
//                System.out.println("k = " + k + " learningRate = " + learningRate);
                NeuralNetwork nn = new NeuralNetwork();
                try {
                    nn.init(_nbInputNodes, _nbHiddenNodes, _nbOutputNodes, this.isRNN, learningRate);
                    nn.kFoldCrossValidation(k, samplesCloned, false);

                    lock.lock();
                    neuralNetworks.add(nn);
                } catch (Exception ex) {
                    Logger.getLogger(DeepLearning.class.getName()).log(Level.SEVERE, null, ex);
                } finally {
                    lock.unlock();
                }

                if (learningRate >= 0.8f && k >= 10) {
                    isRunning = false;
                }
            } while (isRunning);
        }

    }
}
