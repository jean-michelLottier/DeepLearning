/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.jml.nn;

import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author gt902715 <jean-michel.lottier@alten.com>
 * @param <T>
 */
public class Sample<T> {

	private List<T> values;
	private List<T> desiredOutputs;
	private List<T> outputs;
	private boolean example = false;

	public Sample(Sample sample) {
		this.values = new ArrayList<>(sample.getValues());
		this.desiredOutputs = new ArrayList<>(sample.getDesiredOutputs());
		this.outputs = sample.getOutputs() == null ? null : new ArrayList<>(sample.getOutputs());
		this.example = sample.isExample();
	}

	public Sample(List<T> values) {
		this.values = values;
		this.example = false;
	}

	public Sample(List<T> values, List<T> desiredOutputs) throws Exception {
		this.values = values;
		this.desiredOutputs = desiredOutputs;
		this.example = true;
	}

	public List<T> getValues() {
		return values;
	}

	public void setValues(List<T> values) {
		this.values = values;
	}

	public List<T> getDesiredOutputs() {
		return desiredOutputs;
	}

	public void setDesiredOutputs(List<T> desiredOutputs) {
		this.desiredOutputs = desiredOutputs;
		this.example = true;
	}

	public void removeDesiredOutputs() {
		this.desiredOutputs = new ArrayList<>();
		this.example = false;
	}

	public boolean isExample() {
		return example;
	}

	public void setExample(boolean example) {
		this.example = example;
	}

	public List<T> getOutputs() {
		return outputs;
	}

	public void setOutputs(List<T> outputs) {
		this.outputs = outputs;
	}
}
