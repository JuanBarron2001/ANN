import java.util.Random;
import java.util.Arrays;
import java.util.concurrent.TimeUnit;
public class ANN
{

	private static float sigmoid(float x)
	{
		return (float)(1 / (1 + Math.exp(-x)));
	}

	private static float sigmoid_derivative(float x)
	{
		return sigmoid(x)*(1 - sigmoid(x));
	}

	private static float[][] dot(float[][] a, float[][] b)
	{
		float[][] dot = new float[a.length][b[0].length];
		for(int row = 0; row < dot.length; row++)
		{
			for(int col = 0; col < dot[0].length; col++)
			{
				for(int i = 0; i < a[0].length; i++)
				{
					dot[row][col] += a[row][i] * b[i][col];
				}
			}
		}
		return dot;
	}

	private static float[][] add(float[][] a, float b)
	{
		float[][] sum = new float[a.length][1];
		for(int i = 0; i < a.length; i++)
		{
			sum[i][0] = a[i][0] + b;
		}
		return sum;
	}

	private static float[][] sigmoid(float[][] a)
	{
		float[][] sig = new float[a.length][1];
		for(int i = 0; i < a.length; i++)
		{
			sig[i][0] = sigmoid(a[i][0]);
		}

		return sig;
	}

	private static float[][] sub(float[][] a, float[][] b)
	{
		float sub[][] = new float[a.length][1];
		for(int i = 0; i < a.length; i++)
		{
			sub[i][0] = a[i][0] - b[i][0];
		}
		return sub;
	}

	private static float[][] sigmoid_derivative(float[][] a)
	{
		float[][] sig = new float[a.length][1];
                for(int i = 0; i < a.length; i++)
                {
                        sig[i][0] = sigmoid_derivative(a[i][0]);
                }

                return sig;
	}

	public static float[][] mult(float[][] a, float[][] b)
	{
		float[][] mult = new float[a.length][1];
		for(int i = 0; i < a.length; i++)
		{
			mult[i][0] = a[i][0] * b[i][0];
		}
		return mult;
	}

	private static float[][] transpose(float[][] a)
	{
		float[][] trans = new float[a[0].length][a.length];
		for(int r = 0; r < trans.length; r++)
		{
			for(int c = 0; c < trans[0].length; c++)
			{
				trans[r][c] = a[c][r];
			}
		}
		return trans;
	}

	private static float[][] mult(float a, float[][] b)
	{
		float[][] mult = new float[b.length][1];
                for(int i = 0; i < mult.length; i++)
                {
                        for(int j = 0; j < mult[0].length; j++)
			{
				mult[i][j] = b[i][j] * a;
			}
                }
                return mult;
	}

	private static float sum(float[][] a)
	{
		float sum = 0;
		for(int i = 0; i < a.length; i++)
		{
			for(int j = 0; j < a[0].length; j++)
			{
				sum = sum + a[i][j];
			}
		}
		return sum;
	}	

	public static void main(String[] args) throws Exception
	{
		float[][] input_set = {	
					{0,1,0},
					{0,0,1},
					{1,0,0},
					{1,1,0},
					{1,1,1},
					{0,1,1},
					{0,1,0}	
						};
		float[][] labels = {
					{1},
					{0},
					{0},
					{1},
					{1},
					{0},
					{1}
						};

		Random random = new Random(42);

		float[][] weights = {
					{random.nextFloat()},
					{random.nextFloat()},
					{random.nextFloat()}
				};

		System.out.println(Arrays.deepToString(weights));

		float[] bias = {random.nextFloat()};

		System.out.println(bias[0]);

		float lr = .05f;
		
		for(int epoch = 1; epoch <= 25000;epoch++)
		{
			float[][] inputs = input_set.clone();
			float[][] XW = add(dot(inputs,weights), bias[0]);
			float[][] Z = sigmoid(XW);
			float[][] error = sub(Z, labels);
			System.out.println(sum(error));
			float[][] dcost = error.clone();
			float[][] dpred = sigmoid_derivative(Z);
			float[][] z_del = mult(dcost,dpred);
			
			inputs = transpose(input_set);
			
			weights = sub(weights, mult(lr,dot(inputs, z_del)));

			for(float[] arr: z_del)
			{
				for(float num: arr)
				{
					bias[0] = bias[0] - lr * num;
				}
			}
			TimeUnit.SECONDS.sleep(1);

		}

		float[][] single_pt = {{1,0,0}};
		float[][] result = sigmoid(add(dot(single_pt, weights),bias[0]));
		System.out.println(Arrays.deepToString(result));
		Syste.out.println("Hello World");


	}
}
