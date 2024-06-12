import streamlit as st
import random
import torch
import load

def main():
    st.title("图像生成任务")

    y=False

    with st.sidebar:
        n_T=st.slider('请选择生成步数：', min_value=10, max_value=1000)
        user_input = st.number_input("请输入想要生成的数字（0-9）:", step=1,
                                     min_value=0, max_value=9)
        st.text("n_T:" + str(n_T))
        if st.button("确定"):
            st.text("用户消息: " + str(user_input))
            y=True

    if st.button("随机生成图片"):
        x=random.randint(0, 9)
        st.text("用户消息: " + str(x))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        target_class = x  # 生成的数字
        load.load_model_and_generate_images('./data/diffusion_outputs10/model_9.pth', target_class, n_sample=1,
                                       device=device,
                                       guide_w=0.0)

        st.image('generated_images.png', use_column_width=True)

    if(y):
        # st.image('./data/image_ep10_w0.5.png', use_column_width=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        target_class = user_input  # 生成的数字
        load.load_model_and_generate_images('./data/diffusion_outputs10/model_9.pth', target_class, n_sample=1,
                                       device=device,
                                       guide_w=0.0)

        st.image('generated_images.png', use_column_width=True)



if __name__ == "__main__":
    main()
