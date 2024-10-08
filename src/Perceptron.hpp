/// \file

#ifndef PERCEPTRON_INCLUDED
#define PERCEPTRON_INCLUDED

#include <iostream>
#include <cstdlib>
#include "../lib/Eigen/Dense"

namespace neuralnetwork
{
  
  enum InOutType
  {
    DYNAMIC = -1
  };
  
  /// \brief Les entrees et sorties du Perceptron sont un simple vecteur Eigen
  template<typename t, int INPUT_SIZE>
  using InOut = Eigen::Matrix<t, INPUT_SIZE, 1>;
  
  /// \brief On definit dans ce namespace quelques fonctions d'activations, on peut tout de meme en definir ailleurs l'important etant qu'elles aient un unique argument de type double et qu'elles renvoient un double
  namespace activation
  {
    double SIGMOID(double x)
    {
      return 1.0/(1 + exp(-x));
      
    }
    
    double D_SIGMOID(double x)
    {
      double res = exp(x)/pow(exp(x)+1, 2);
      if (res!=res)
      {
        res = 0.0001;
      }
      return res;
    }
    double AFFINE(double x)
    {
      return x;
    }
    double D_AFFINE(double x)
    {
      return 1.0;
    }
  }
  

  /// \brief Classe Perceptron
  /// \class Perceptron
  /// \tparam DEPTH nombres de couches du perceptron sans compter la couche d'entrée
  template<std::size_t DEPTH>
  class Perceptron
  {

    public:
      
      /// \brief Constructeur, les poids du reseau sont initialises entre deux bornes de maniere aleatoire, on definit egalement la taille des couches et leurs fonctions d'activation
      /// \param min Valeur minimal d'un poids du reseau
      /// \param max Valeur maximal d'un poids du reseau
      /// \param args Trois arguments pour definie une couche, le premier est le nombre de neurone de la couche, le deuxieme est est la fonction d'activation, le troisieme est la derivee de la fonction d'activation. Les trois premiers arguments definissent la premiere couche cachee, les trois suivants la deuxieme couche cachee, etc... la fonction d'activation et sa derivee doivent etre des pointeurs sur fonction qui prennent un double et retourne un double
      template <typename ...Args>
      Perceptron(double min, double max, Args... args);
      
      /// \brief Initialisation aleatoire des poids du reseau entre deux bornes
      /// \param min Valeur minimal d'un poids du reseau
      /// \param max Valeur maximal d'un poids du reseau
      /// \return Rien
      void initWeights(double min, double max);
      
      /// \brief Accesseur des poids du reseau
      /// \param couch La couche a laquelle appartient le neurone qui a le poids sur une de ses entrees
      /// \param neuron Le neurone de la couche
      /// \param previous Le neurone de la couche precedente
      /// \return Accesseur sur le poids
      double& weight(std::size_t couch, std::size_t neuron, std::size_t previous);
      
      /// \brief Nombre de poids de chaque neurone d'une couche
      /// \param couch La couche en question
      /// \return le nombre de poids pour un neurone de la couche en question
      std::size_t weights_count(std::size_t couch);
      
      /// \brief Profondeur du reseau
      /// \return La profondeur du reseau sans prendre en compte la couche d'entree
      std::size_t depth();
      
      /// \brief Nombre de neurones d'une couche donnee
      /// \param couch La couche en question
      /// \return Nombre de neurones de la couche en question
      std::size_t couch_size(std::size_t couch);
      
      //Eigen::Matrix<double, Eigen::Dynamic, 1> agreg(std::size_t couch);
      /// \brief Les dernieres agregations enregistrees pour une couche donnee (methode principalement utile au debugage)
      /// \param couch La couche en question
      /// \return une entree-sortie de neurones qui comporte toutes les dernieres agregation enregistrees de la couche en question
      neuralnetwork::InOut<double,neuralnetwork::InOutType::DYNAMIC> agreg(std::size_t couch);
      
      //Eigen::Matrix<double, Eigen::Dynamic, 1> activation(std::size_t couch);
      /// \brief Les dernieres activations enregistrees pour une couche donnee (methode principalement utile au debugage)
      /// \param couch La couche en question
      /// \return une entree-sortie de neurones qui comporte toutes les dernieres activations enregistrees de la couche en question
      neuralnetwork::InOut<double,neuralnetwork::InOutType::DYNAMIC> activation(std::size_t couch);
      
      
      /// \brief Ecrit la matrice des poids d'une couche, une ligne correspond a un neurone, les poids sont dans l'odre des entrees provenant de la couche precedente et les agregations et activations de cette couche (methode principalement utile au debugage)
      /// \param out reference sur le flux de sortie ou l'on veut afficher les informations
      /// \param i La couche en question
      /// \return Rien
      void printMatrixCouch(std::ostream& out, std::size_t i);
      
      /// \brief Evalue une entree avec le reseau
      /// \param in L'entree a evaluer
      /// \param out La sortie du reseau
      /// \param saveAgreg Si place a true, les agregations des differentes couches seront enregistrees
      /// \param saveActivation Si place a true, les activations des differentes couches seront enregistrees
      /// \tparam tin type d'entree du reseau (a priori un double)
      /// \tparam INPUT_SIZE taille d'entree du reseau
      /// \tparam tout type des sorties du reseau (a priori un double)
      /// \tparam OUTPUT_SIZE taille de sortie du reseau
      /// \return Rien
      template<typename tin, int INPUT_SIZE, typename tout, int OUTPUT_SIZE>
      void feedForward(neuralnetwork::InOut<tin, INPUT_SIZE>& in, neuralnetwork::InOut<tout, OUTPUT_SIZE>& out, bool saveAgreg, bool saveActivation);
     
      /// \brief Applique la fonction d'activation sur l'agregation calulee d'une couche
      /// \param in L'aggregation calculee de la couche fi
      /// \param fi La couche en question
      /// \return Rien
      /// \tparam tin type de l'agreation (a priori un double)
      /// \tparam INPUT_SIZE taille de la couche
      /// \tparam tout type de l'activation (a priori un double)
      /// \tparam OUTPUT_SIZE taille de la couche
      /// \todo possibilite de se passer de in si feedForward enregistre systematiquement l'agregation
      template<typename t, int INPUT_SIZE>
      void applyActivation(neuralnetwork::InOut<t, INPUT_SIZE>& in, std::size_t fi);
      
      /// \brief Applique l'algorithme de retropropagation du gradient avec un jeu d'exemples afin d'entrainer le reseau
      /// \param ins Tableau des exemples d'entrees du reseau
      /// \param outs Tableau des sorties attendues pour chaque exemple de ins (organises dans le meme ordre)
      /// \param n Nombre d'exemples
      /// \param step Pas d'apprentissage
      /// \return Rien
      /// \tparam tin type des entrees (a priori double)
      /// \tparam INPUT_SIZE taille d'entree du reseau
      /// \tparam tout type des sorties du reseau (a priori double)
      /// \tparam OUTPUT_SIZE taille de sortie du reseau
      template<typename tin, int INPUT_SIZE, typename tout, int OUTPUT_SIZE>
      void backpropagation(neuralnetwork::InOut<tin, INPUT_SIZE>* ins, neuralnetwork::InOut<tout, OUTPUT_SIZE>* outs, std::size_t n, double step);

    private:
      
      /// \brief Methode utilitaire pour le constructeur, on definit egalement la taille des couches et leurs fonctions d'activation. Cette methode parcourt recursivement toutes les couches
      /// \param n Couche a parametrer
      /// \param input_size Nombres de poids pour les neurones de cette couche, egal au nombre de neurones de la couche precedente
      /// \param neurons Taille de cette couche en nombres de neurones
      /// \param f Fonction d'activation de cette couche
      /// \param df Derivee de la fonction d'activation de cette couche
      /// \param args Parametres pour les couches suivantes
      /// \return Rien
      template<typename... Args>
      void init(int n, int input_size, int neurons, double (*f)(double), double (*df)(double), Args ...args);
      
      /// \brief Dernier appel de init pour la derniere couche a parametrer
      void init(int n, int input_size, int neurons, double (*f)(double), double (*df)(double));
      
      /// \brief tableau des fonctions d'activations, une par couche
      double (*m_array_f[DEPTH])(double);
      
      /// \brief tableau des derivees des fonctions d'activations, une par couche
      double (*m_array_df[DEPTH])(double);
      
      /// \brief profondeur du reseau
      const std::size_t m_size = DEPTH;
     
      /// \brief tableau des matrices des couches. Une ligne represente les poids d'un neurone, l'element i de la ligne j represente le poids reliant le sortie du neurones i de la couche precedente avec le neurone j de la couche actuelle
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> m_array_matrix[DEPTH];
      
      /// \brief tableau des agrgations
      Eigen::Matrix<double, Eigen::Dynamic, 1> m_array_agreg[DEPTH];
      
      /// \brief tableau des activations
      Eigen::Matrix<double, Eigen::Dynamic, 1> m_array_act[DEPTH];
      
      /// \brief tableau des erreurs
      Eigen::Matrix<double, Eigen::Dynamic, 1> m_array_err[DEPTH];
      
  };
}

/// \fn std::ostream& operator<<(std::ostream& out, neuralnetwork::Perceptron<DEPTH> p)
/// \brief Affiche les matrices de poids du reseau et les agregations et activations de chaque couche (methode principalement utile au debugage)
/// \param out reference sur le flux de sortie ou l'on veut afficher les informations
/// \param p Perceptron a afficher
/// \return out avec de nouvelles informations concernant le reseau dedans
/// \warning La fonction est incoherente, une partie des informations est ecrite dans out et l'autre dans std::cout. Cette fonction sert principalement au debugage
template<std::size_t DEPTH>
std::ostream& operator<<(std::ostream& out, neuralnetwork::Perceptron<DEPTH> p)
{
  out << DEPTH <<" couches" << std::endl;
  for(int i = 0; i < DEPTH; ++i)
  {
    out << "Couche " << i <<std::endl;
    p.printMatrixCouch(out, i);
    //getchar();
    out << std::endl;
    out << "Agregation:" << std::endl << p.agreg(i);
    //getchar();
    out << std::endl;
    out << "Activation:" << std::endl << p.activation(i);
    //getchar();
    out << std::endl;
  }

  return out;
}

template<std::size_t DEPTH>
template <typename ...Args>
neuralnetwork::Perceptron<DEPTH>::Perceptron(double min, double max, Args... args)
{
  init(0, args...);
  initWeights(min, max);
}


template<std::size_t DEPTH>
template<typename... Args>
void neuralnetwork::Perceptron<DEPTH>::init(int n, int input_size, int neurons, double (*f)(double), double (*df)(double), Args ...args)
{
  //On dimensionne la matrice representant la couche
  m_array_matrix[n].resize(neurons, input_size);

  //On donne la bonne fonction et la bonne derivee a la couche
  m_array_f[n] = f;
  m_array_df[n] = df;
  if(n < DEPTH )
  {
    init(n+1, neurons, args...);
  }
}

template<std::size_t DEPTH>
void neuralnetwork::Perceptron<DEPTH>::init(int n, int input_size, int neurons, double (*f)(double), double (*df)(double))
{
  m_array_matrix[n].resize(neurons, input_size);
  m_array_f[n] = f;
  m_array_df[n] = df;
}

template<std::size_t DEPTH>
void neuralnetwork::Perceptron<DEPTH>::initWeights(double min, double max)
{
  for(int i = 0; i<depth() ;++i)
  {
    for(int j = 0; j<couch_size(i); ++j)
    {
      for(int k = 0; k<weights_count(i); ++k)
      {
        weight(i, j, k) = (std::rand()/(static_cast<double>(RAND_MAX) + 1.0)) * (max - min) + min;
      }
    }
    
  }
} 

template<std::size_t DEPTH>
double& neuralnetwork::Perceptron<DEPTH>::weight(std::size_t couch, std::size_t neuron, std::size_t previous)
{
  return m_array_matrix[couch](neuron, previous);
}
 
template<std::size_t DEPTH>
std::size_t neuralnetwork::Perceptron<DEPTH>::weights_count(std::size_t couch)
{
  return m_array_matrix[couch].cols();
}

template<std::size_t DEPTH>
std::size_t neuralnetwork::Perceptron<DEPTH>::depth()
{
  return m_size;
}

template<std::size_t DEPTH>
std::size_t neuralnetwork::Perceptron<DEPTH>::couch_size(std::size_t couch)
{
  return m_array_matrix[couch].rows();
}

template<std::size_t DEPTH>
void neuralnetwork::Perceptron<DEPTH>::printMatrixCouch(std::ostream& out, std::size_t i)
{
  std::cout << m_array_matrix[i];
}

template<std::size_t DEPTH>
Eigen::Matrix<double, Eigen::Dynamic, 1> neuralnetwork::Perceptron<DEPTH>::agreg(std::size_t couch)
{
  return m_array_agreg[couch];
}
  
template<std::size_t DEPTH>
Eigen::Matrix<double, Eigen::Dynamic, 1> neuralnetwork::Perceptron<DEPTH>::activation(std::size_t couch)
{
  return m_array_act[couch];
}

template<std::size_t DEPTH>
template<typename t, int INPUT_SIZE>
void neuralnetwork::Perceptron<DEPTH>::applyActivation(neuralnetwork::InOut<t, INPUT_SIZE>& in, std::size_t fi)
{
  for(int i = 0; i < in.size(); ++i)
  {
    in(i) = m_array_f[fi](in(i));
  }
}

template<std::size_t DEPTH>
template<typename tin, int INPUT_SIZE, typename tout, int OUTPUT_SIZE>
void neuralnetwork::Perceptron<DEPTH>::feedForward(neuralnetwork::InOut<tin, INPUT_SIZE>& in, neuralnetwork::InOut<tout, OUTPUT_SIZE>& out, bool saveAgreg, bool saveActivation)
{
  InOut<tin, neuralnetwork::InOutType::DYNAMIC> tmp;
  
  //feed forward pour le premiere couche
  tmp = m_array_matrix[0]*in;
  
  if(saveAgreg)
  {
    m_array_agreg[0] = tmp;
  }
  applyActivation(tmp, 0);
  if(saveActivation)
  {
    m_array_act[0] = tmp;
  }

  //feed forward pour les autres couches
  for(int i = 1; i < DEPTH; ++i)
  {
    tmp = m_array_matrix[i] * tmp;
    if(saveAgreg)
    {
      m_array_agreg[i] = tmp;
    }
    applyActivation(tmp, i);
    if(saveActivation)
    {
      m_array_act[i] = tmp;
    }
  }

  out = tmp;
}

template<std::size_t DEPTH>
template<typename tin, int INPUT_SIZE, typename tout, int OUTPUT_SIZE>
void neuralnetwork::Perceptron<DEPTH>::backpropagation(neuralnetwork::InOut<tin, INPUT_SIZE>* ins, neuralnetwork::InOut<tout, OUTPUT_SIZE>* outs, std::size_t n, double step)
{
  neuralnetwork::InOut<tout, Eigen::Dynamic> out;
  neuralnetwork::InOut<tout, Eigen::Dynamic> err;
  
  //Pour chaque exemple d'entrainement
  for(int i = 0; i < n; ++i)
  {
    //On evalue un exemple avec le reseau
    feedForward(ins[i], out, true, true);
    
    //On calcul l'erreur en sortie
    
    //difference entre la sortie desiree et la sortie attendue
    err = outs[i] - out;
    //df(agreg)*(sortie_desiree - sortie_attendue)
    for(int j = 0; j < err.size(); ++j)
    {
      err(j) = m_array_df[DEPTH - 1](m_array_agreg[DEPTH - 1](j)) * err(j);
    }
    m_array_err[DEPTH - 1] = err;

    //Erreur pour les autres couches
    //erreur(j - 1) = df(agreg) * matriceJ.transpose * erreur(j)
    for(int j = DEPTH - 1; j > 0; --j)
    {
      err = m_array_matrix[j].transpose() * m_array_err[j];
      for(int k = 0; k < err.size(); ++k)
      {
        err(k) = m_array_df[j - 1](m_array_agreg[j - 1](k)) * err(k);
      }
      m_array_err[j - 1] = err;
    }

    //TODO sortir le cas de la première couche pour eliminer if_else
    //Mise a jour des poids
    for(int j = 0; j < DEPTH; ++j)
    {
      for(int k = 0; k < couch_size(j); ++k)
      {
        for(int l = 0; l < weights_count(j); ++l)
        {
          if(j > 0)
          {
            weight(j, k, l) += step * m_array_err[j](k) * m_array_act[j - 1](l);
          }
          //Pour la premiere couche on prend le vecteur d'entree
          else
          {
            weight(j, k, l) += step * m_array_err[j](k) * ins[i](l);
          }
        }
      }
    }

  }
}

#endif
