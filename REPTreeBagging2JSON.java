import weka.classifiers.*;
import weka.classifiers.trees.*;
import weka.classifiers.meta.Bagging;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import java.util.ArrayList;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;

import java.lang.reflect.*;

public class REPTreeBagging2JSON {
	public static void main(String[] args) throws Exception {
		String filepath = args[0] + ".model";
		Bagging obj = (Bagging) SerializationHelper.read(filepath);
		// obj.getTree(0).setNumDecimalPlaces(17);
		PrintWriter pw = new PrintWriter(new FileWriter(args[0] + ".json"));
		pw.print("[\n");
		
		Class<?> cls = obj.getClass().getSuperclass().getSuperclass().getSuperclass();
		Field field = cls.getDeclaredField("m_Classifiers");
		field.setAccessible(true);
		Classifier[] rt = (Classifier[]) field.get(obj);
                for (int k = 0; k < rt.length; k++) {
		cls = rt[k].getClass();
		field = cls.getDeclaredField("m_Tree");
		field.setAccessible(true);
		Class<?> Tree = Class.forName("weka.classifiers.trees.REPTree$Tree");
		Object n = Tree.cast(field.get(rt[k]));
		cls = n.getClass();
		Field fFeature = cls.getDeclaredField("m_Attribute");
		fFeature.setAccessible(true);
		Field fThreshold = cls.getDeclaredField("m_SplitPoint");
		fThreshold.setAccessible(true);
		Field fChildren = cls.getDeclaredField("m_Successors");
		fChildren.setAccessible(true);
		Field fClassDistribution = cls.getDeclaredField("m_Distribution");
		fClassDistribution.setAccessible(true);
		ArrayList<Object> q = new ArrayList<>();
		ArrayList<Integer> attribute = new ArrayList<>();
		ArrayList<Double> splitPoint = new ArrayList<>();
		ArrayList<Integer> IDLeft = new ArrayList<>();
		ArrayList<Integer> IDRight = new ArrayList<>();
		int node_count = 0;
		int node_dequeue = 0;
		q.add(n);
		node_count++;
		// System.out.println(obj.getTree(0).toString());
		while (node_dequeue < node_count) {
			n = q.get(node_dequeue);
			if ((int) fFeature.get(n) >= 0) {
				attribute.add((int) fFeature.get(n));
				splitPoint.add((double) fThreshold.get(n));
				IDLeft.add(node_count);
				IDRight.add(node_count+1);
				q.add(((Object[]) fChildren.get(n))[0]);
				q.add(((Object[]) fChildren.get(n))[1]);
				node_count += 2;
			} else {
				attribute.add(-1);
				splitPoint.add(-1.0);
				IDLeft.add(-1);
				IDRight.add(-1);
			}
			node_dequeue++;
		}
		double[] cnt0 = getCount(fFeature, fClassDistribution, q, IDLeft, IDRight, 0);
		double[] cnt1 = getCount(fFeature, fClassDistribution, q, IDLeft, IDRight, 1);
		pw.print("{\"children_left\": [");
		for (int i = 0; i < q.size()-1; i++) {
			pw.print(IDLeft.get(i) + ",");
		}
		pw.print(IDLeft.get(q.size()-1) + "],\n");

		pw.print("\"children_right\": [");
		for (int i = 0; i < q.size()-1; i++) {
			pw.print(IDRight.get(i) + ",");
		}
		pw.print(IDRight.get(q.size()-1) + "],\n");
		pw.print("\"feature\": [");
		for (int i = 0; i < q.size()-1; i++) {
			pw.print(attribute.get(i) + ",");
		}
		pw.print(attribute.get(q.size()-1) + "],\n");
		pw.print("\"threshold\": [");
		for (int i = 0; i < q.size()-1; i++) {
			pw.print(splitPoint.get(i) + ",");
		}
		pw.print(splitPoint.get(q.size()-1) + "],\n");
		pw.print("\"value\": [");
		for (int i = 0; i < q.size()-1; i++) {
			pw.print(cnt1[i] / (cnt0[i] + cnt1[i]) + ",");
		}
		pw.print(cnt1[q.size()-1] / (cnt0[q.size()-1] + cnt1[q.size()-1]) + "],\n");
		pw.print("\"node_sample_weight\": [");
		for (int i = 0; i < q.size()-1; i++) {
			pw.print((cnt0[i] + cnt1[i]) + ",");
		}
		pw.print((cnt0[q.size()-1] + cnt1[q.size()-1]) + "]}" + (k < rt.length-1 ? ",\n" : "\n"));
		}
		pw.print("]\n");
		pw.close();
	}
	private static double[] getCount(Field fFeature, Field fClassDistribution, ArrayList<Object> q, ArrayList<Integer> idL, ArrayList<Integer> idR, int c) throws IllegalArgumentException, IllegalAccessException {
		int sz = q.size();
		double[] ret = new double[sz];
		int i = sz - 1;
		while (i >= 0) {
			Object t = q.get(i);
			if ((int) fFeature.get(t) >= 0) {
				ret[i] = ret[idL.get(i)] + ret[idR.get(i)];
			} else {
				ret[i] = ((double[]) fClassDistribution.get(t))[c];
			}
			i--;
		}
		return ret;
	}
}
