from sqlalchemy import Column, Integer, String, JSON, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Video(Base):
    __tablename__ = "video"
    id = Column(Integer, primary_key=True, autoincrement=True)
    path = Column(String, nullable=False)

    frame = relationship("Frame", back_populates="video")


class Frame(Base):
    __tablename__ = "frame"
    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(
        Integer, ForeignKey("video.id", ondelete="CASCADE"), nullable=False
    )
    timestamp = Column(Integer, nullable=False)
    dogs_num = Column(Integer, nullable=False)

    video = relationship("Video", back_populates="frame")
    label = relationship("Label", back_populates="frame")


class Label(Base):
    __tablename__ = "label"
    id = Column(Integer, primary_key=True, autoincrement=True)
    frame_id = Column(
        Integer, ForeignKey("frame.id", ondelete="CASCADE"), nullable=False
    )
    x1 = Column(Integer, nullable=False)
    y1 = Column(Integer, nullable=False)
    x2 = Column(Integer, nullable=False)
    y2 = Column(Integer, nullable=False)
    class_name = Column(String, nullable=False)

    frame = relationship("Frame", back_populates="label")