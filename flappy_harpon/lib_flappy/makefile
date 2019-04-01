CPP=gcc    #Commande du compilateur
CFLAGS=-O3 -fPIC #Option d'optimisation du programme
# LDFLAGS=-lSDL2 -lSDL2_mixer #Linker
# LDFLAGS=-lSDL -lSDL_mixer -lSDL_image -lSDL_ttf#Linker
LIB=lib_flappy.a #Nom du programme � modifier

OBJS=flappy.o graphique.o game.o tools.o

all: $(LIB)

-include $(OBJS:.o=.d)

lib_flappy.a: $(OBJS)
	ar rcs lib_flappy.a $(OBJS)

%.o: %.c
	$(CPP) -g $(CFLAGS) -c $*.c
	$(CPP) -g -MM $*.c > $*.d

clean:
	rm -fr *.o *.a *.d

mrproper: clean
	rm -fr $(LIB)
