# 3-Layer-Perceptron

## Welcome to Our Multi-Layer Perceptron Project 👩‍💻🧠

Hey there, fellow developer! Are you ready to dive into MLP model? This README will guide you through the setup and workflow for our project, step by step. Whether you're a rookie or a seasoned pro, we've got you covered. Let's do this! 🚀

---

## 1️⃣ Data Preparation: Local Is the Way to Go 📂

Tired of dealing with rigid project setups because of large datasets? No worries — we've made sure you can customize paths and keep things flexible! Here's the process:

1. **Download the Dataset Locally:**  
   Use the `data_download.py` script to download the dataset to your desired location. Pro tip: Customize the download path in `data_download.py` to match your local setup.  
   
2. **Keep Things Consistent:**  
   Update the corresponding paths in **both** `data_preprocess.py` and `main.py` so everything runs smoothly. No loose ends here!  

3. **Run the Preprocessing Script:**  
   Fire up the `data_preprocess.py` script to preprocess the dataset. The processed data will be saved for later use.

4. **Visualize Preprocessing Results:**  
   Check out the preprocessing visualization results stored in the `data_preprocess_visualization` folder. It's always cool to see stuff come to life, isn't it? 😎

---

## 2️⃣ Baseline Model: Simple Yet Powerful 💡

Now that your data is ready, let's get our hands dirty with the baseline model: a neat three-layer MLP. Here's how you can get started:  

1. **Train and Test the Model:**  
   Run `main.py`, and it will handle training, testing, and plotting the **accuracy-and-loss curves** for you.  

2. **Change Hyperparameters:**  
   Need to experiment with different setups? Head over to `config.py` and tweak the hyperparameters — who knows, you might discover the next big thing in AI!  

3. **Save Models Smartly:**  
   In `main.py`, customize the model save path to something like `best_model_relu` or `best_model_sigmoid` — this way, you avoid clutter and stay organized.  

4. **Visualize Parameters:**  
   Run `plot.py` to visualize the model parameters. The results will magically appear in the `visualization` folder. Because data science without visuals is like pizza without cheese. 🍕

---

## 3️⃣ Attention Model: Focus Where It Matters 👀  

Feeling adventurous? Then it's time to dive into the world of **attention mechanisms**, one of the coolest tricks in modern machine learning. Here's how:  

1. **Enter the Attention Zone:**  
   Navigate to the `LAB1_attention` subfolder. No detours necessary.  

2. **Similar Workflow, New Powers:**  
   The process here is almost identical to the **baseline model** reproduction steps above. But hey, attention will bring out the magic!

---

## Troubleshooting 🛠️  

We all know that sometimes things don't go as planned. If you hit a roadblock, don't sweat it — mistakes are just opportunities to learn! Feel free to reach out to us at **22307140084@m.fudan.edu.cn**, and we'll get back to you faster than a GPU can train a model.  If you have any idea about the attention mechanism, welcome to communicate with us as well!

---

## Ready, Set, Code! 🎉  

Now you’re all set to conquer this project of MLP. No matter if you're preprocessing data, fine-tuning hyperparameters, or exploring attention mechanisms, this project has something for everyone. And hey, don’t forget to have fun — because coding is always better with a smile! 😄

Happy coding! 💻✨
