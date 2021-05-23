import pygame
import random
import neat
import os
import pickle

from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_ESCAPE,
    KEYDOWN,
    QUIT,
    RLEACCEL
)

generation = 0
global_best_score = -1

SCREEN_WIDTH = 500
SCREEN_HEIGHT = 700

PLAYER_WIDTH = 40
PLAYER_HEIGHT = 40
PLAYER_SPEED = 5

ENEMY_WIDTH = 10
ENEMY_HEIGHT = 20
ENEMY_SPEED = 6

GREY = (211,211,211)
BLACK = (8,8,8)
BLUE = (0,255,9)
RED = (240,0,0)
WHITE = (255,255,255)
SKY_BLUE = (135,206,235)

PLAYER_SPRITE_IMAGE_PATH = "airplane_sprite.png"
ENEMY_SPRITE_IMAGE_PATH = "missile_sprite.png"
NEAT_CONFIG_PATH = "config-feedforward.cfg"


class Player(pygame.sprite.Sprite):
    """
    Player Class
    """

    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.speedx = PLAYER_SPEED
        self.speedy = PLAYER_SPEED
        self.surf = pygame.image.load(PLAYER_SPRITE_IMAGE_PATH).convert()
        self.surf.set_colorkey((255,255,255),RLEACCEL)
        self.rect = self.surf.get_rect(center=(SCREEN_WIDTH/2,SCREEN_HEIGHT-PLAYER_HEIGHT))
        self.is_alive = True
        self.score = 0

    def update(self,k_input):
        if k_input=="left":
            self.rect.move_ip(-self.speedx,0)
        elif k_input=="right":
            self.rect.move_ip(self.speedx,0)
        else:
            pass
            
        if self.rect.left <= 0:
            self.rect.left = 0
        if self.rect.right >= SCREEN_WIDTH:
            self.rect.right = SCREEN_WIDTH
        if self.rect.bottom >= SCREEN_HEIGHT:
            self.rect.bottom = SCREEN_HEIGHT
        if self.rect.top <=0:
            self.rect.top = 0
            
    def get_data(self,enemies):
        """
        Takes enemies sprite group as input and returns input values to be used in neural network

        :param enemies:
        :return: tuple of values indicating number enemies in different area w.r.t player
        """
        frontfar = pygame.Rect(self.rect.left,SCREEN_HEIGHT-PLAYER_HEIGHT*10,PLAYER_WIDTH,PLAYER_HEIGHT*7)
        leftfar = pygame.Rect(self.rect.left-PLAYER_WIDTH,SCREEN_HEIGHT-PLAYER_HEIGHT*10,PLAYER_WIDTH,PLAYER_HEIGHT*7)
        rightfar = pygame.Rect(self.rect.left+PLAYER_WIDTH,SCREEN_HEIGHT-PLAYER_HEIGHT*10,PLAYER_WIDTH,PLAYER_HEIGHT*7)
        frontnear = pygame.Rect(self.rect.left,SCREEN_HEIGHT-PLAYER_HEIGHT*3,PLAYER_WIDTH,PLAYER_HEIGHT*2)
        leftnear = pygame.Rect(self.rect.left-PLAYER_WIDTH,SCREEN_HEIGHT-PLAYER_HEIGHT*3,PLAYER_WIDTH,PLAYER_HEIGHT*2)
        rightnear = pygame.Rect(self.rect.left+PLAYER_WIDTH,SCREEN_HEIGHT-PLAYER_HEIGHT*3,PLAYER_WIDTH,PLAYER_HEIGHT*2)
        
        radarfrontfar = [sprite for sprite in enemies if sprite.rect.colliderect(frontfar)]
        radarleftfar = [sprite for sprite in enemies if sprite.rect.colliderect(leftfar)]
        radarrightfar = [sprite for sprite in enemies if sprite.rect.colliderect(rightfar)]
        radarfrontnear = [sprite for sprite in enemies if sprite.rect.colliderect(frontnear)]
        radarleftnear = [sprite for sprite in enemies if sprite.rect.colliderect(leftnear)] 
        radarrightnear = [sprite for sprite in enemies if sprite.rect.colliderect(rightnear)]  

        return (len(radarfrontfar),len(radarleftfar),len(radarrightfar),len(radarfrontnear),len(radarleftnear),len(radarrightnear))
    
    def get_alive(self):
        return self.is_alive


class Enemy(pygame.sprite.Sprite):
    """
    Enemy class
    """

    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.surf = pygame.image.load(ENEMY_SPRITE_IMAGE_PATH).convert()
        self.surf.set_colorkey((255,255,255),RLEACCEL)
        self.rect = self.surf.get_rect(center=(random.randint(0,SCREEN_WIDTH),random.randint(-5,10)))
        self.speed = ENEMY_SPEED
        
    def update(self):
        
        self.rect.move_ip(0,self.speed)
        if self.rect.top > SCREEN_HEIGHT:
            self.kill()


def run_game(genomes,config):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))

    all_sprites = pygame.sprite.Group()
    enemies = pygame.sprite.Group()
    clouds = pygame.sprite.Group()

    ADDENEMY = pygame.USEREVENT + 1 
    pygame.time.set_timer(ADDENEMY,350) 
    
    clock = pygame.time.Clock()

    running = True

    players = []
    nets = []
    ge = []

    for genome_id,genome in genomes:
        player = Player()
        genome.fitness=0
        ge.append(genome)
        players.append(Player())
        network = neat.nn.FeedForwardNetwork.create(genome,config)
        nets.append(network)

    global generation
    global global_best_score
    generation+=1
    generation_font = pygame.font.SysFont("Arial", 20)

    while running:
        for event in pygame.event.get():
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False

            elif event.type == ADDENEMY:
                enemy = Enemy()
                all_sprites.add(enemy)
                enemies.add(enemy)

        # For each player nn get input to it

        for i,player in enumerate(players):
            output = nets[i].activate(player.get_data(enemies))
        
            if output[0] < 0:
                k_input = "left"
                
            elif output[0] > 0:
                k_input = "right"
                
            else:
                k_input = "none"
                
            player.update(k_input)
            
        remain_players = 0    

        # Check if player collides with enemy group

        for i,player in enumerate(players):
            if pygame.sprite.spritecollideany(player,enemies):
                player.is_alive = False
                player.kill()
                ge[i].fitness -= 100
                
            if player.get_alive():
                remain_players += 1
                all_sprites.add(player)
                ge[i].fitness += 1
                players[i].score += 0.1
                
        best_score = -999
        best_index = -999
        
        for i,player in enumerate(players):
            if player.score > best_score:
                best_score = player.score
                best_index = i
        
        if best_score> global_best_score:
            global_best_score = best_score
            
        # Giving extra reward if breaks best score
        ge[best_index].fitness += 2
        
        if remain_players==0:
            running = False    
            
        screen.fill(SKY_BLUE)
        clouds.update()
        enemies.update()
        for elm in all_sprites:
            screen.blit(elm.surf,elm.rect)

        text1 = generation_font.render("Generation : " + str(generation), True, BLACK)
        text1_rect = text1.get_rect()
        text1_rect.center = (SCREEN_WIDTH/2, 50)
        
        text2 = generation_font.render("Alive : " + str(remain_players), True, BLACK)
        text2_rect = text2.get_rect()
        text2_rect.center = (SCREEN_WIDTH/2, 80)
        
        text3 = generation_font.render("Score&Index : " + str(round(best_score))+"_"+str(best_index), True, BLACK)
        text3_rect = text3.get_rect()
        text3_rect.center = (SCREEN_WIDTH/2, 110)
        
        text4 = generation_font.render("Global best score : " + str(round(global_best_score)), True, BLACK)
        text4_rect = text4.get_rect()
        text4_rect.center = (SCREEN_WIDTH/2, 140)
        
        screen.blit(text1, text1_rect)    
        screen.blit(text2, text2_rect) 
        screen.blit(text3, text3_rect)  
        screen.blit(text4, text4_rect)          
            
        pygame.display.flip()
        clock.tick(30)
        

def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, 
    neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file
    )
    
    p = neat.Population(config)
    
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Saving the winner as pickle object later to be used
    winner = p.run(run_game,60)
    winner_path = os.path.join(local_dir,"winner.pk1")
    with open(winner_path,"wb") as f:
        pickle.dump(winner,f)
    
    print(winner)

    print("Best score:",global_best_score)


if __name__ == '__main__':

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, NEAT_CONFIG_PATH)
    run(config_path)