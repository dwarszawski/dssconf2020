if __name__ == '__main__':
    import dill

    file_path=""
    with open("explainer.dill", "wb") as file:
        dill.dump(explainer, file)
        file_path = file.name