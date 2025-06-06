library(Rmisc);
library(ggplot2);
library(data.table);

# initialize the data frame
df = data.frame(col1 = numeric(), col2 = character(), col2 = numeric(), stringsAsFactors = FALSE);

df[1, ] <- c(10, "divergence", 125.785);
df[2, ] <- c(100, "divergence", 131.836);
df[3, ] <- c(1000, "divergence", 131.015);
df[4, ] <- c(10000, "divergence", 124.944);
df[5, ] <- c(100000, "divergence", 145.382);
df[6, ] <- c(1000000, "divergence", 249.056);
df[7, ] <- c(10000000, "divergence", 468.877);
df[8, ] <- c(100000000, "divergence", 1914.08);

df[9, ] <- c(10, "no_divergence", 129.302);
df[10, ] <- c(100, "no_divergence", 135.303);
df[11, ] <- c(1000, "no_divergence", 124.874);
df[12, ] <- c(10000, "no_divergence", 125.184);
df[13, ] <- c(100000, "no_divergence", 136.105);
df[14, ] <- c(1000000, "no_divergence", 238.506);
df[15, ] <- c(10000000, "no_divergence", 460.351);
df[16, ] <- c(100000000, "no_divergence", 1847.53);

# set the names of the columns
colnames(df) <- c("size", "type", "latency");

# convert the columns in numeric data type
df$size <- as.numeric(as.character(df$size));
df$latency <- as.numeric(as.character(df$latency));

print(df)

# plotting
plot <- ggplot(df, aes(x = size, y = latency, color = type, group = type)) +
    theme_bw() +
    theme(plot.title = element_text(hjust = 0.5)) +
    xlab("vector size (elements)") +
    ylab("kernel latency (usec)") +
    geom_point(size=4.4) +
    geom_line(linewidth=0.8) +
    ggtitle("DivergenceTest") +
    theme(legend.position="bottom") +
    theme(legend.title= element_blank(), legend.text=element_text(size=14,face="bold")) +
    theme(axis.text=element_text(size=14), axis.title=element_text(size=16,face="bold"), plot.title=element_text(size=18,face="bold")) +
    theme(aspect.ratio=0.5);

# saving the plot in a pdf file
ggsave("divergence.pdf", plot);
